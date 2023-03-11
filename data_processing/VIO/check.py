import os
import os.path as osp

trajectory_dir = '/home/jianrenw/Research/data/jianrenw'

demos = os.listdir(trajectory_dir)
for demo in demos:
    frames = sorted(os.listdir(osp.join(trajectory_dir, demo)))
    if len(frames) < 30:
        print(demo)