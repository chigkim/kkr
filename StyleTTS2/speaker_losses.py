import torch
from torch import nn
import torch.nn.functional as F


class SpeakerClassifier(nn.Module):
    def __init__(self, style_dim, num_speakers, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or style_dim
        self.net = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_speakers),
        )

    def forward(self, x):
        return self.net(x)


class SpeakerCenters(nn.Module):
    def __init__(self, num_speakers, style_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_speakers, style_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    @property
    def weight(self):
        return self.embedding.weight

    def forward(self, speaker_ids):
        return self.embedding(speaker_ids)


def speaker_separation_loss(
    style,
    speaker_ids,
    classifier=None,
    centers=None,
    lambda_ce=0.0,
    lambda_center=0.0,
    lambda_margin=0.0,
    margin=1.0,
):
    loss = style.new_tensor(0.0)
    ce = style.new_tensor(0.0)
    center = style.new_tensor(0.0)
    margin_loss = style.new_tensor(0.0)

    if classifier is not None and lambda_ce > 0:
        ce = F.cross_entropy(classifier(style), speaker_ids.long())
        loss = loss + lambda_ce * ce

    if centers is not None and lambda_center > 0:
        target = centers(speaker_ids.long())
        center = F.mse_loss(style, target)
        loss = loss + lambda_center * center

    # DDP wraps modules; attribute access must go through .module.
    center_weight = getattr(centers, 'module', centers).weight if centers is not None else None
    if center_weight is not None and lambda_margin > 0 and center_weight.shape[0] > 1:
        normalized = F.normalize(center_weight, dim=-1)
        distances = torch.cdist(normalized, normalized, p=2)
        mask = ~torch.eye(distances.shape[0], dtype=torch.bool, device=distances.device)
        margin_loss = F.relu(margin - distances[mask]).mean()
        loss = loss + lambda_margin * margin_loss

    return loss, ce.detach(), center.detach(), margin_loss.detach()
