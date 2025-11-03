# DiffRL 修复说明

## 问题总结

原始问题:
1. CUDA 扩展编译失败 - 无效的 `-Z` 编译选项
2. CUDA 11.5 + g++ 11 不兼容 - `std::function` 参数包错误
3. PyTorch 1.11 (CUDA 11.3) 不支持 RTX 4090 (sm_89)
4. NetworkX 2.2 使用过时的 `np.int` API

## 已实施的修复

### 1. 修复编译选项 (`dflex/dflex/adjoint.py`)
- 移除无效的 `-Z` 选项
- 修改为 `-O2 -DNDEBUG -std=c++14`
- 更新 CUDA 架构为 `compute_86`(向后兼容 sm_89)

### 2. 安装 g++-10
```bash
sudo apt-get install g++-10
```

### 3. 升级 PyTorch
从 1.11.0 (CUDA 11.3) 升级到 2.0.1 (CUDA 11.8):
```bash
conda activate shac
pip uninstall -y torch torchvision torchaudio
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

### 4. 升级 NetworkX
从 2.2 升级到 3.1 以修复 numpy 兼容性问题:
```bash
pip install --upgrade networkx
```

## 使用方法

### 方式 1: 使用辅助脚本
```bash
cd /home/chunleli/Dev/DiffRL/examples
/home/chunleli/Dev/DiffRL/run_with_g++10.sh python test_env.py --env AntEnv
```

### 方式 2: 手动设置环境变量
```bash
cd /home/chunleli/Dev/DiffRL/examples
conda activate shac
export CC=gcc-10
export CXX=g++-10
export TORCH_CUDA_ARCH_LIST="8.6"
python test_env.py --env AntEnv
```

## 验证

成功运行输出应显示:
```
fps =  ~17000
mean reward =  ~1281
Finish Successfully
```

## 技术细节

- **系统 CUDA**: 11.5 (nvcc)
- **PyTorch CUDA**: 11.8 (运行时库)
- **GPU**: NVIDIA GeForce RTX 4090 (sm_89, compute capability 8.9)
- **编译架构**: compute_86 (向后兼容,可在 sm_89 上运行)
- **编译器**: g++-10 (与 CUDA 11.5 兼容)

## 注意事项

1. **每次运行需要设置环境变量** - 使用 `run_with_g++10.sh` 脚本自动设置
2. **首次编译耗时较长** - 后续使用缓存会更快
3. **CUDA 架构限制** - 系统 CUDA 11.5 不支持 compute_89,使用 compute_86 作为替代

## 日期
2025-10-29
