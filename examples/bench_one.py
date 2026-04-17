"""One-shot SHAC throughput probe.

Usage:
    python bench_one.py --cfg ./cfg/shac/snu_humanoid.yaml \
                        --num-actors 128 --max-epochs 8 --device cuda:1

Loads the given SHAC config, overrides num_actors / max_epochs, runs
SHAC.train(). The per-epoch line printed by SHAC includes "fps total {N}"
which the driver script greps to compute throughput.
"""
import argparse
import os
import sys
import time

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

import torch
import yaml

import envs  # noqa: F401  -- registers env classes
import algorithms.shac as shac
from utils.common import seeding, get_time_stamp  # noqa: F401


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', required=True)
    p.add_argument('--num-actors', type=int, required=True)
    p.add_argument('--max-epochs', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default='cuda:1')
    p.add_argument('--logdir', type=str, default='./logs/_bench')
    args = p.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    cfg['params']['config']['num_actors'] = args.num_actors
    cfg['params']['config']['max_epochs'] = args.max_epochs
    cfg['params']['config']['save_interval'] = 0

    logdir = os.path.join(args.logdir, f'na{args.num_actors}_{int(time.time())}')
    cfg['params']['general'] = {
        'seed': args.seed,
        'device': torch.device(args.device),
        'render': False,
        'logdir': logdir,
        'train': True,
        'play': False,
        'test': False,
        'checkpoint': 'Base',
        'no_time_stamp': True,
    }

    print(f'[BENCH] num_actors={args.num_actors} max_epochs={args.max_epochs} '
          f'device={args.device}')
    t0 = time.time()
    trainer = shac.SHAC(cfg)
    t_init = time.time() - t0
    print(f'[BENCH] init_time_s={t_init:.2f}')

    t0 = time.time()
    trainer.train()
    t_train = time.time() - t0
    print(f'[BENCH] train_time_s={t_train:.2f}')


if __name__ == '__main__':
    main()
