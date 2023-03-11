import os
import os.path as osp
import numpy as np
import pickle5 as pickle
import csv
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F


class GripperDataset(Dataset):

    def __init__(self, valid_videos, train_list, val_list, mode, task):
        super().__init__()
        self.valid_videos = valid_videos

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
            self.color_t = transforms.Compose(
                [transforms.Resize((240, 240)),
                 transforms.CenterCrop(224)])

    def __len__(self):
        return len(self.valid_videos)

    def __getitem__(self, idx):
        frames = sorted(os.listdir(self.valid_videos[idx]))[:-1]
        label_file = sorted(os.listdir(self.valid_videos[idx]))[-1]
        points = np.sort(np.random.choice(len(frames), 1, replace=False))
        current_point = points[0]

        with open(osp.join(self.valid_videos[idx], label_file),
                  newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                label = int(row[0])

        with open(osp.join(self.valid_videos[idx], frames[current_point]),
                  'rb') as handle:
            frame_data = pickle.load(handle)
            T_stbt0 = np.eye(4)
            T_stbt0[:3, :3] = R.from_quat(frame_data['r']).as_matrix()
            T_stbt0[:3, 3] = frame_data['p']
            color_image = frame_data['rgb']
            color_image = self.color_t(
                torch.from_numpy(color_image / 255.0).permute((2, 0, 1)))

        if current_point >= label:
            action = torch.Tensor([1.])
        else:
            action = torch.Tensor([0.])

        return color_image.float() * 255.0, action