import pickle5 as pickle
import glob
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R


parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, help="Data Root Dir.")
parser.add_argument("--task", type=str, help="Task Name.")

args = parser.parse_args()
videos = glob.glob(args.root_dir + "/" + args.task + "/*")

all_actions = []

for i, video in enumerate(videos):
    print(video)
    all_frames = sorted(glob.glob(video + "/*"))[::5]
    frames = []
    for frame in all_frames:
        if '.csv' not in frame:
            frames.append(frame)
    with open(frames[0], 'rb') as handle:
        frame_data = pickle.load(handle)
        T_stbt0 = np.eye(4)
        T_stbt0[:3, :3] = R.from_quat(frame_data['r']).as_matrix()
        T_stbt0[:3, 3] = frame_data['p']

    for frame in frames[1:]:
        with open(frame, 'rb') as handle:
            frame_data = pickle.load(handle)
            T_stbt1 = np.eye(4)
            T_stbt1[:3, :3] = R.from_quat(frame_data['r']).as_matrix()
            T_stbt1[:3, 3] = frame_data['p']

        T_bt0_bt1 = np.linalg.inv(T_stbt0) @ T_stbt1
        if args.task in ['pushing', 'pick_and_place', 'door_opening']:
            scale = np.linalg.norm(T_bt0_bt1[:3, 3])
            if scale > 0.005:
                gt_action = T_bt0_bt1[:3, 3] / scale
                all_actions.append(gt_action)
        elif args.task in ['knob_turning']:
            all_actions.append(T_bt0_bt1)
        T_stbt0 = T_stbt1

with open('data_loader/{}_{}.pickle'.format("action", args.task), 'wb') as handle:
    pickle.dump(all_actions, handle, protocol=pickle.HIGHEST_PROTOCOL)
