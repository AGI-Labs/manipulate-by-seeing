import os
import os.path as osp
import numpy as np
import pickle5 as pickle
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
import cv2
from torch import linalg as LA

class PassiveDataset(Dataset):

    def __init__(self, valid_videos, train_list, val_list, task, mode):
        super().__init__()
        self.valid_videos = valid_videos
        self.task = task

        with open('data_loader/{}_{}.pickle'.format("action", task),
                  'rb') as handle:
            self.random_actions = pickle.load(handle)

        if mode == 'train':
            self.valid_videos = self.valid_videos[train_list].tolist()
            self.color_t = transforms.Compose([
                transforms.Resize((240, 240)),
                transforms.RandomGrayscale(p=0.05),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                transforms.RandomCrop(224)
            ])
        elif mode == 'val':
            self.valid_videos = self.valid_videos[val_list].tolist()
            self.color_t = transforms.Compose([
                transforms.Resize((240, 240)),
                transforms.CenterCrop(224)
            ])

    def __len__(self):
        return len(self.valid_videos)

    def __getitem__(self, idx):
        all_frames = sorted(os.listdir(self.valid_videos[idx]))[::5]
        frames = []
        for frame in all_frames:
            if '.csv' not in frame:
                frames.append(frame)
        points = np.sort(np.random.choice(len(frames), 2, replace=False))

        current_point = points[0]
        target_point = points[1]

        with open(osp.join(self.valid_videos[idx], frames[current_point]),
                  'rb') as handle:
            frame_data = pickle.load(handle)
            T_stbt0 = np.eye(4)
            T_stbt0[:3, :3] = R.from_quat(frame_data['r']).as_matrix()
            T_stbt0[:3, 3] = frame_data['p']
            try:
                current_image = cv2.imdecode(frame_data['bgr_enc'],
                                            cv2.IMREAD_COLOR)[:, :, ::-1]
            except KeyError:
                try:
                    current_image = frame_data['bgr'][:, :, ::-1]
                except KeyError:
                    current_image = frame_data['rgb'][:, :, ::-1]
            current_image = self.color_t(
                torch.from_numpy(current_image / 255.).permute((2, 0, 1)))

        with open(osp.join(self.valid_videos[idx], frames[current_point + 1]),
                  'rb') as handle:
            frame_data = pickle.load(handle)
            T_stbt1 = np.eye(4)
            T_stbt1[:3, :3] = R.from_quat(frame_data['r']).as_matrix()
            T_stbt1[:3, 3] = frame_data['p']
            try:
                next_image = cv2.imdecode(frame_data['bgr_enc'],
                                            cv2.IMREAD_COLOR)[:, :, ::-1]
            except KeyError:
                try:
                    next_image = frame_data['bgr'][:, :, ::-1]
                except KeyError:
                    next_image = frame_data['rgb'][:, :, ::-1]
            next_image = self.color_t(
                torch.from_numpy(next_image / 255.).permute((2, 0, 1)))

        with open(osp.join(self.valid_videos[idx], frames[target_point]),
                  'rb') as handle:
            frame_data = pickle.load(handle)
            try:
                target_image = cv2.imdecode(frame_data['bgr_enc'],
                                            cv2.IMREAD_COLOR)[:, :, ::-1]
            except KeyError:
                try:
                    target_image = frame_data['bgr'][:, :, ::-1]
                except KeyError:
                    target_image = frame_data['rgb'][:, :, ::-1]
            target_image = self.color_t(
                torch.from_numpy(target_image / 255.).permute((2, 0, 1)))

        T_bt0_bt1 = np.linalg.inv(T_stbt0) @ T_stbt1
        np.random.shuffle(self.random_actions)
        candidate_actions = np.stack(self.random_actions)

        if self.task in ['pushing', 'pick_and_place', 'door_opening']:
            gt_action = T_bt0_bt1[:3, 3] / np.linalg.norm(T_bt0_bt1[:3, 3])
        elif self.task in ['knob_turning']:
            gt_action = T_bt0_bt1.reshape(-1)[:12]
            candidate_actions = candidate_actions.reshape(-1, 16)[:, :12]
        
        gt_action = torch.from_numpy(gt_action)
        candidate_actions = torch.from_numpy(candidate_actions)

        best_candidate_index = torch.argmin(
            LA.norm(gt_action - candidate_actions, dim=1))

        return current_image.float() * 255.0, next_image.float() * 255.0, target_image.float() * 255.0, gt_action.float(
        ), candidate_actions.float(), best_candidate_index
