import torch
import torch.nn as nn
import torch.nn.functional as F


class BCLoss(nn.Module):

    def __init__(self):
        super(BCLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')
        self.direction = nn.CosineSimilarity(dim=1)

    def forward(self, translation_predictions, rotation_predictions,
                translation_gts, rotation_gts):

        translation_loss = self.mse(translation_gts, translation_predictions)
        rotation_loss = self.mse(rotation_gts, rotation_predictions)
        direction_loss = torch.mean(
            1 - self.direction(translation_predictions, translation_gts))
        return translation_loss + direction_loss + rotation_loss, translation_loss, rotation_loss

class IBCLoss(nn.Module):

    def __init__(self):
        super(IBCLoss, self).__init__()
        self.cse = nn.CrossEntropyLoss()

    def forward(self, similarity, bext_action_index):
        return self.cse(similarity, bext_action_index)

class SpatialLoss(nn.Module):

    def __init__(self):
        super(SpatialLoss, self).__init__()
        self.cse = nn.CrossEntropyLoss()

    def forward(self, similarity, bext_action_index):
        return self.cse(similarity, bext_action_index)

class DynamicsLoss(nn.Module):

    def __init__(self):
        super(DynamicsLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, predict_features, next_features):
        return self.mse(predict_features, next_features)

class DistanceLoss(nn.Module):

    def __init__(self):
        super(DistanceLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, predict_distance, gt_distance):
        return self.mse(predict_distance, gt_distance)

class GripperLoss(nn.Module):

    def __init__(self):
        super(GripperLoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, predict_action, gt_action):
        return self.bce(predict_action, gt_action)