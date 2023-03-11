import torch
import torch.nn as nn
import torch.nn.functional as F
from r3m import load_r3m


class gripper_action(nn.Module):

    def __init__(self):
        super().__init__()
        self.image_feature = load_r3m("resnet18")
        self.projection = nn.Sequential(nn.ReLU(),
                                        nn.Linear(512, 1, bias=True),
                                        nn.Sigmoid())

    def forward(self, current_image):
        image_feature = self.image_feature(current_image).view(-1, 512)
        action = self.projection(image_feature)
        return action


