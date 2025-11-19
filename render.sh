cd /home/chunleli/Dev/DiffRL/examples 
# python train_shac.py --cfg ./cfg/shac/ant.yaml --checkpoint ./logs/Ant/shac/40/best_policy.pt --play --render --device cuda:0
python train_shac.py --cfg ./cfg/shac/snu_humanoid.yaml --checkpoint ./logs/SNUHumanoid/shac/0/best_policy.pt --play --render --device cuda:0
