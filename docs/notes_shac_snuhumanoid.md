# DiffRL · SHAC + SNUHumanoid 学习笔记

本文整理对本仓库 SHAC 算法、SNUHumanoid 环境、可微仿真机制的理解与实测结果。
代码引用格式 `path/to/file.py:LINE`。

---

## 1. 并行环境数量(num_actors)

### 1.1 是 GPU 并行,不是 CPU 多进程
- env 张量全部在 CUDA 上,默认 `device='cuda:0'`(`envs/snu_humanoid.py:32`)
- 物理模型整体搬到 GPU:`self.builder.finalize(self.device)`(`envs/snu_humanoid.py:193`)
- dflex 是单卡 batched 可微仿真器,所有 env 在同一 CUDA kernel 内 batch 执行
- **没有** `SubprocVecEnv` / 多进程;加 CPU 核数无效
- 多 GPU(DDP)当前**不支持**,需自己改

### 1.2 配置位置
`examples/cfg/<algo>/<env>.yaml` 里的 `num_actors`(等价于 `num_envs`)。
SNUHumanoid + SHAC 在 `examples/cfg/shac/snu_humanoid.yaml:37`。
`player.num_actors`(line 43)只在 `--play`/`--test` 时覆盖,见 `examples/train_shac.py:96`。

### 1.3 实测吞吐(SNUHumanoid + SHAC, RTX 4090 24GB, 取 iter 3-6 平均)

| num_actors | fps total | 加速比 | 备注 |
|---|---|---|---|
| 64(默认) | 1,614 | 1.00× | |
| 128 | 2,337 | 1.45× | |
| 256 | 2,899 | 1.80× | |
| 512 | 5,707 | 3.53× | 此处加速比明显跳升(SM 启动开销摊销) |
| 1024 | 10,218 | **6.33×** | 推荐 |
| 1536 | 13,918 | **8.62×** | 接近显存极限,稳定可跑 |
| 2048 | — OOM — | — | 初始化阶段已占 ~22GB |

`init_time` 随 num_actors 线性涨(64→3s, 1024→65s, 1536→76s),正式训练 2000 epoch 时可忽略。

仅看 `nvidia-smi dmon` 的 SM% 会被低估并行收益(小 batch 时 kernel launch 开销 + low occupancy 也能让 SM 显示忙)。**`fps total` 才是可信指标**。

### 1.4 提速优先级
1. `num_actors` 64 → 1024(最直接,~6×)
2. SM 饱和后才考虑:
   - `MM_caching_frequency: 8 → 16`(`snu_humanoid.yaml:6`)
   - `sim_substeps: 48 → 32`(`envs/snu_humanoid.py:97`)
   - `critic_iterations: 16 → 8`(`snu_humanoid.yaml:27`)
3. 改大 batch 后,`actor_learning_rate / critic_learning_rate` 可能需相应放大或加 warmup
4. 不要做:Python 多进程、AMP/`torch.compile`(可微仿真对数值精度敏感,梯度易炸)

### 1.5 临时基准脚本
`examples/bench_one.py` —— 接受 `--num-actors --max-epochs --device`,跑短训练并打印 `fps total`。
注意:dflex JIT 编译的 CUDA kernel 默认绑 cuda:0,直接传 `--device cuda:1` 会 illegal memory access。
绕过方式:`CUDA_VISIBLE_DEVICES=1 python ... --device cuda:0`。

---

## 2. SNUHumanoid 资产解析

### 2.1 资产文件(`envs/assets/snu/`)
- `human.xml` —— 骨骼/关节/刚体,SNU 自定义 XML 格式(非 MJCF/URDF)
- `muscle284.xml` —— 284 块肌肉,每块含多个 `<Waypoint>`
- `OBJ/*.obj`、`OBJ/*.usd` —— 渲染网格(物理上不使用,只用 box 近似)
- `motion/*.bvh` —— **未被代码引用**,见 §3

### 2.2 入口(`envs/snu_humanoid.py:130-167`)
```python
for i in range(self.num_environments):
    skeleton = lu.Skeleton(asset_path, muscle_path, self.builder, self.filter, ...)
    self.builder.joint_q[skeleton.coord_start + 1] = 1.0  # 抬到地面 1m
    self.skeletons.append(skeleton)
```
**每个并行 env 创建独立 `Skeleton`**,依次追加到同一 `df.sim.ModelBuilder`,
最终 `joint_q` 长度 = `单体 q 维度 × num_envs`(line 170-171)。

### 2.3 `Skeleton` 类(`utils/load_utils.py:502`)
构造时调用两件事:
- `parse_skeleton(skeleton_file, builder, filter)`(line 537)
- `parse_muscles(muscle_file, builder)`(line 667)

### 2.4 `parse_skeleton`(line 537-665)
对 XML 中每个 `<Node>`:
| XML 字段 | 用途 |
|---|---|
| `parent` | 父子链(`node_map`/`xform_map`) |
| `<Body>` 的 `mass/size/obj/Transformation` | box 形状 + `density = mass / (x·y·z)`(line 586) |
| `<Joint>` 的 `type/axis/lower/upper/Transformation` | 类型映射 `Ball/Revolute/Prismatic/Free/Fixed → df.JOINT_*`(line 548) |

**filter 机制是 SNUHumanoid 的关键减负手段**(line 624):
```python
self.filter = { "Pelvis", "FemurR", "TibiaR", ..., "FootPinkyL" }  # snu_humanoid.py:34
if len(filter) == 0 or name in filter:
    builder.add_link(...); builder.add_shape_box(...)
```
原始 XML 是全身骨架,但只把**下半身 11 个关节**加为可仿真 link。
其它(臂、躯干、手指)只在 `node_map` 里记名字+变换,不参与物理。
所以动作维 = 18(`num_joint_qd - 6`,见 `snu_humanoid.py:52`)。

刚度/阻尼按 `mass_scale = body_mass/15` 缩放(line 588-589, 640-643)。

### 2.5 `parse_muscles`(line 667-718)
对每个 `<Unit>` 读 `f0/lm/lt/lmax/pen_angle`,遍历每个 `<Waypoint>`:
```python
way_link = self.node_map[way_bone]
joint_X_s = self.xform_map[way_bone]
way_loc = df.transform_point(df.transform_inverse(joint_X_s), way_loc)  # 世界系→关节局部系
m.bones.append(way_link); m.points.append(way_loc)

if not incomplete:    # 任何 waypoint 落在被 filter 掉的 link 上 → 整块肌肉丢弃
    builder.add_muscle(m.bones, m.points, f0=..., lm=..., lt=..., lmax=..., pen=...)
```
原 284 块经 filter 后剩 **152** 块(`snu_humanoid.py:48`),对应 actor 输出维度。

### 2.6 finalize → GPU
`snu_humanoid.py:193`:
```python
self.model = self.builder.finalize(self.device)
```
之后 `joint_q / joint_qd / muscle_activation` 都以 `[num_envs, ...]` 形状直接在 GPU 上 batch 仿真。

### 2.7 数据流总览
```
human.xml ──┐
            ├──► Skeleton.parse_skeleton ──► builder.add_link / add_shape_box (×11 下半身)
filter  ────┘
                                                 ▼
muscle284.xml ──► Skeleton.parse_muscles ──► builder.add_muscle (×152, filter 后)
                                                 ▼
       (上述循环执行 num_envs 次,共享同一 builder)
                                                 ▼
                                  builder.finalize(device) → GPU Model
```

### 2.8 想加新身体部位
1. 在 `snu_humanoid.py:34` 的 `filter` 里加新 Node 名
2. 同步改 `num_joint_q / num_joint_qd / num_muscles`(line 44-48)
3. 相应调整 obs/act 维度(line 52-56)

---

## 3. Motion 文件未被使用

### 3.1 资产里有 motion
`envs/assets/snu/motion/`:`backflip.bvh, balance.bvh, cart.bvh, dance.bvh, kick.bvh, pirouette.bvh, run.bvh, walk.bvh`
标准 BVH 格式,根节点 `Character1_Hips`,层级与 `human.xml` 里 `<Joint bvh="Character1_*">` 完全对应 ——
**资产是为 motion tracking 准备好的**。

### 3.2 但代码不加载
全仓 `\.bvh|BVH|motion/` grep 在 `*.py` 里 **0 命中**。
`snu_humanoid.py` 也搜不到 `motion / track / imitat / reference` 任何字眼。

### 3.3 实际任务是向前跑(纯 locomotion)
`calculateReward`(`envs/snu_humanoid.py:411`):
- `up_reward` = 0.1 · 躯干向上分量
- `heading_reward` = 朝向远端目标
- `height_reward` = 关于身高 0.46m 的 piecewise(别摔倒)
- `progress_reward` = 距远端 target 的位移
- `action_penalty` = -0.001 · action²

`self.targets = [10000, 0, 0]`(line 119),目标是 X 方向 10km 远的点,
等价于"让人形向前跑得越远越好"。`obs_buf` 也无任何参考 phase 或目标关节角。

### 3.4 motion 文件来源
源自 SNU(Seoul National University)Lee 等人 *Scalable Muscle-Actuated Human Simulation and Control*(SIGGRAPH 2019)。
原项目就是肌肉驱动 motion tracking,BVH 是数据集遗留;NVIDIA 抽骨架+肌肉做 DiffRL benchmark 时一起拷过来了。

### 3.5 想做 motion tracking 需要补
- BVH 解析器(`bvh-converter` / `bvhtoolbox` / 自写)
- retarget 每帧关节角到 11 个保留 link
- `calculateReward` 加 `pose_reward = exp(-||q - q_ref(t)||²)`,obs 里加 phase `t`
- 决定动作空间用关节力矩还是 152 块肌肉激活

---

## 4. SHAC 如何利用可微信息

### 4.1 与传统 RL 的根本区别

| | PPO / SAC | **SHAC** | BPTT |
|---|---|---|---|
| 策略梯度 | 得分函数 ∇log π · Â | **∂(累计奖励)/∂θ 解析回传** | 同 SHAC,horizon=整条 traj |
| 用 env 梯度 | 否 | **是** | 是 |
| critic | 是 | **是,负责 horizon 之外** | 否 |
| Horizon | 数百~上千 | **短(`steps_num=32`)** | 整条 |

PPO 把 env 当黑盒;SHAC 把 env 当**可微函数** `s_{t+1}, r_t = f(s_t, a_t)`,
像训普通可微网络一样回传梯度。

### 4.2 前向:32 步深的可微计算图
`algorithms/shac.py:191-198`:
```python
for i in range(self.steps_num):                     # steps_num = 32
    actions = self.actor(obs, deterministic=False)  # actor MLP
    obs, rew, done, _ = self.env.step(torch.tanh(actions))   # 可微仿真 + 奖励
    rew_acc[i+1] = rew_acc[i] + gamma * rew
```
`env.step` 内部 `envs/snu_humanoid.py:282`:
```python
self.state = self.integrator.forward(self.model, self.state, self.sim_dt, self.sim_substeps, ...)
```
通过 `dflex/sim.py:2086` 的 `SimulateFunc(torch.autograd.Function)`:
- **forward**:跑 48 个 substep 半隐式欧拉,记录在 `df.Tape`(line 2107-2123)
- **backward**:`tape.replay()` 倒着跑回去,得 ∂s_{t+1}/∂(s_t, a_t, model_params)(line 2126-2154)

`obs` 和 `rew` 都是带梯度张量,reward 函数(`calculateReward`)是纯 PyTorch op,梯度天然贯通。

### 4.3 Actor loss = 累计奖励 + 末端 critic
`shac.py:248-251`:
```python
# 中途 env 终止
actor_loss += (-rew_acc[i+1, done_ids] - γ·γ_t·V_target(s_{i+1})).sum()
# 最后一步,所有 env 截断
actor_loss += (-rew_acc[T, :] - γ·γ_t·V_target(s_T, :)).sum()
```
等价数学形式:
```
J(θ) = E[ Σ_{t=0..T-1} γ^t · r_t(θ) + γ^T · V_φ(s_T(θ)) ]
```
其中 `r_t(θ)` 和 `s_T(θ)` 都通过可微仿真显式依赖 θ。

### 4.4 Backward:一次 `loss.backward()` 推到 actor
`shac.py:411`:`actor_loss.backward()`

链式法则路径:
```
∂J/∂θ_actor
   ← ∂J/∂rew_t          (跨 32 步累计)
        ← ∂rew_t/∂s_t   (来自 reward 函数解析微分)
        ← ∂s_t/∂s_{t-1} (来自 dflex SimulateFunc.backward = tape.replay)
              ← ... 链式回 s_0
        ← ∂s_t/∂a_{t-1}
              ← ∂a_{t-1}/∂θ_actor  (来自 actor MLP)
   ← ∂J/∂V_φ(s_T)·∂V_φ/∂s_T·(∂s_T/∂θ_actor)   (critic 提供"未来梯度")
```
**这条链穿透整个物理仿真器**:关节力 → 加速度 → 速度 → 位置 → 接触力 → 下一帧观测。
PPO 永远拿不到。

### 4.5 Critic 的角色
为什么不微一整条 episode(纯 BPTT)?
- 物理混沌 + 接触不连续 → 长 horizon 梯度爆炸/消失/NaN
- BPTT 算力/显存随 T 线性涨

SHAC 折中:**只微 32 步**,之后用 critic `V_φ(s_T)` 估计后续期望累计奖励,
其解析梯度 `∂V_φ/∂s_T` 接到链尾 —— critic 是 horizon truncation 的"梯度延伸器"。

Critic 训练侧(`shac.py:349-369`)是**完全经典的 TD(λ)**:
```python
target_values = TD_lambda(rew_buf, next_values, γ, λ=0.95)   # 只读不微
critic_loss = ((V_φ(obs) - target_values) ** 2).mean()        # MSE
```
不需要仿真梯度。

### 4.6 工程稳定性技巧
| 问题 | 解决方案 | 位置 |
|---|---|---|
| 跨 SHAC 迭代 BPTT 链无限增长 | `initialize_trajectory()` 切断梯度 | `snu_humanoid.py:368`, `shac.py:184` |
| 接触不连续致 NaN 梯度 | `register_hook(nan_to_num)` | `snu_humanoid.py:260-272` |
| 梯度尺度不稳 | `clip_grad_norm_(grad_norm=1.0)` | `shac.py:417` |
| Critic bootstrap 噪声 | Polyak target critic α=0.995 | `shac.py:534-538` |
| `done` 时丢弃 `next_values` | line 232 | `shac.py:227-239` |
| 异常大值立刻 ValueError | 显式 sanity check | `shac.py:241, 421` |

### 4.7 数据流图
```
            ┌──────── steps_num = 32 ────────┐
   π_θ → a_0 → SIM → s_1, r_0 → π_θ → a_1 → SIM → ... → s_32, r_31
    │              │                                       │
    │              └─ 自动微分: SimulateFunc.backward      │
    │                                                      ▼
    │                                                 V_φ(s_32)   ← critic 提供"未来梯度"
    │                                                      │
    └────── ∂J/∂θ ← 链式回传穿过仿真器 ←────────────────────┘
                                       │
                            actor_optimizer.step()

(critic 这边: target = TD-λ on (r, V_φ-target) → MSE → critic_optimizer.step())
```

### 4.8 为什么 SHAC 比 PPO 快
- **梯度信噪比**:得分函数估计方差 O(1),需海量 sample 平均;解析梯度方差几乎为 0
- **样本效率**:论文报告同 wall-clock 下快 5-10×
- **代价**:必须有可微仿真器,且仿真器对接触/弹性的处理决定梯度质量;
  不可微环境(MuJoCo/Atari/真机)用不了

---

## 附录:常用 cheatsheet

```bash
# 训练入口(run.sh 已配置)
cd /home/chunleli/Dev/DiffRL/examples
python train_script.py --env SNUHumanoid --algo shac --num-seeds 5

# 临时改 num_actors 跑基准
CUDA_VISIBLE_DEVICES=1 python bench_one.py \
    --cfg ./cfg/shac/snu_humanoid.yaml \
    --num-actors 1024 --max-epochs 6 --device cuda:0

# 监控 GPU
nvidia-smi dmon -s u -i 0
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# tensorboard
tensorboard --logdir=./logs/ --port=6008
```

关键文件速查:
- 算法: `algorithms/shac.py`
- 环境: `envs/snu_humanoid.py`、`envs/dflex_env.py`
- 资产解析: `utils/load_utils.py`(`Skeleton`/`MuscleUnit`)
- 可微仿真: `dflex/dflex/sim.py`(`SimulateFunc`、`SemiImplicitIntegrator`)
- 配置: `examples/cfg/shac/snu_humanoid.yaml`
