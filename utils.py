import numpy as np
import random
import torch
from torch import nn


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor_emb, positive_emb, negative_emb):
        distance_pos = torch.norm(anchor_emb - positive_emb, dim=1, p=2)
        distance_neg = torch.norm(anchor_emb - negative_emb, dim=1, p=2)
        loss = torch.mean(torch.clamp(distance_pos - distance_neg + self.margin, min=0))
        return loss


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device('cpu')
