# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import argparse
import runpy
import shlex

configs = {'Ant': 'ant.yaml', 'CartPole': 'cartpole_swing_up.yaml', 'Hopper': 'hopper.yaml', 'Cheetah': 'cheetah.yaml', 'Humanoid': 'humanoid.yaml', 'SNUHumanoid': 'snu_humanoid.yaml'}

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='Ant', choices=['Ant', 'CartPole', 'Hopper', 'Cheetah', 'Humanoid', 'SNUHumanoid'])
parser.add_argument('--algo', type=str, default='shac', choices=['shac', 'ppo', 'sac', 'bptt'])
parser.add_argument('--num-seeds', type=int, default=5)
parser.add_argument('--save-dir', type=str, default='./logs/')

args = parser.parse_args()

''' generate seeds '''
seeds = []
for i in range(args.num_seeds):
    seeds.append(i * 10)

''' generate commands '''
commands = []
for i in range(len(seeds)):
    seed = seeds[i]
    save_dir = os.path.join(args.save_dir, args.env, args.algo, str(seed))
    config_path = os.path.join('./cfg', args.algo, configs[args.env])

    if args.algo == 'shac':
        script_name = 'train_shac.py'
    elif args.algo == 'ppo' or args.algo == 'sac':
        script_name = 'train_rl.py'
    elif args.algo == 'bptt':
        script_name = 'train_bptt.py'
    else:
        raise NotImplementedError

    
    cmd = 'python {} '\
        '--cfg {} '\
        '--seed {} '\
        '--logdir {} '\
            .format(script_name, config_path, seed, save_dir)

    commands.append(cmd)

for command in commands:
    # 允许在同一 Python 进程内执行子脚本, 便于调试器(step into)
    # 示例 command: "python train_shac.py --cfg ... --seed ... --logdir ..."
    args_list = shlex.split(command)

    # 提取脚本路径与参数(跳过前缀的 "python")
    if len(args_list) < 2:
        print(f"Skip invalid command: {command}")
        continue

    script_name = args_list[1]
    script_path = script_name
    if not os.path.isabs(script_name):
        # 与本文件同目录下的训练脚本
        script_path = os.path.join(os.path.dirname(__file__), script_name)

    # 设置 sys.argv 以模拟命令行参数
    argv_backup = sys.argv[:]
    sys.argv = [script_name] + args_list[2:]

    print(f"\n=== Running in-process: {script_name} {' '.join(sys.argv[1:])} ===")
    try:
        # 在当前进程中执行, debug 时可进入子脚本
        runpy.run_path(script_path, run_name="__main__")
    except SystemExit as e:
        # 捕获 argparse 或脚本内显式退出, 不中断后续任务
        code = e.code if isinstance(e.code, int) else 0
        print(f"{script_name} exited with code {code}")
    finally:
        # 恢复调用方 argv
        sys.argv = argv_backup

print("\nAll runs finished.")
