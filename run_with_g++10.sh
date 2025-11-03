#!/bin/bash

# 使用 g++-10 解决 CUDA 11.5 + g++ 11 兼容性问题
export CC=gcc-10
export CXX=g++-10

# 设置 CUDA 架构 (8.6 兼容 RTX 4090/sm_89,因为系统 CUDA 11.5 不支持 compute_89)
export TORCH_CUDA_ARCH_LIST="8.6"

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate shac

# 运行传入的命令
"$@"
