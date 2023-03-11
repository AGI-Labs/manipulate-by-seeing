import torch
import torch.nn as nn
import torch.nn.functional as F
from r3m import load_r3m


class dynamics_model(nn.Module):

    def __init__(self, action_dim):
        super().__init__()
        self.prediction = nn.Sequential(nn.Linear(128 + action_dim, 128, bias=True),
                                        nn.ReLU(),
                                        nn.Linear(128, 128, bias=True),
                                        nn.ReLU(),
                                        nn.Linear(128, 128, bias=True))

    def forward(self, state, action):
        prediction = self.prediction(torch.cat((state, action), -1))
        return prediction / torch.linalg.norm(
            prediction, axis=1, keepdim=True)


class state_feature(nn.Module):

    def __init__(self):
        super().__init__()
        self.image_feature = load_r3m("resnet18")
        self.projection = nn.Sequential(nn.ReLU(),
                                        nn.Linear(512, 128, bias=True))

    def forward(self, current_image):
        image_feature = self.image_feature(current_image).view(-1, 512)
        state = self.projection(image_feature)
        return state / torch.linalg.norm(
            state, axis=1, keepdim=True)
