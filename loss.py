"""Loss functions used by EFAN.

CAAHL (Cohesion-Aware Adaptive Hybrid Loss) dynamically balances InfoNCE
(contrastive) and Circle (margin-based) optimization according to batch-wise
feature cohesion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE(nn.Module):
    """Symmetric InfoNCE loss computed on the cross-modal similarity matrix."""

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.loss_function = loss_function
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)

        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        logits_per_image2 = logits_per_image1.T

        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)

        loss = (self.loss_function(logits_per_image1, labels) +
                self.loss_function(logits_per_image2, labels)) / 2

        return loss


class CircleLoss(nn.Module):
    """Symmetric Circle Loss for contrastive learning.

    Reference: "Circle Loss: A Unified Perspective of Pair Similarity
    Optimization" (https://arxiv.org/abs/2002.10857).
    """

    def __init__(self,
                 m: float = 0.25,
                 gamma: float = 256.0,
                 delta_p=0.75,
                 delta_n=0.25,
                 reduction: str = 'mean',
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.m = m
        self.gamma = gamma
        self.delta_p = delta_p
        self.delta_n = delta_n
        self.reduction = reduction
        self.device = device

        assert delta_p > delta_n, f"delta_p ({delta_p}) must be greater than delta_n ({delta_n})"
        assert reduction in ['mean', 'sum'], "reduction must be 'mean' or 'sum'"

    def forward(self, image_features1, image_features2, logit_scale):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)

        logits_per_image1 = image_features1 @ image_features2.T
        logits_per_image2 = logits_per_image1.T

        loss1 = self._circle_loss_single(logits_per_image1, logit_scale)
        loss2 = self._circle_loss_single(logits_per_image2, logit_scale)

        return (loss1 + loss2) / 2

    def _circle_loss_single(self, similarity_matrix, logit_scale):
        batch_size = similarity_matrix.size(0)

        pos_mask = torch.eye(batch_size, dtype=torch.bool, device=self.device)
        neg_mask = ~pos_mask

        pos_similarities = similarity_matrix[pos_mask].unsqueeze(1)  # [B, 1]
        neg_similarities = similarity_matrix[neg_mask].view(batch_size, batch_size - 1)

        alpha_p = torch.relu(self.delta_p - pos_similarities.detach())
        alpha_n = torch.relu(neg_similarities.detach() - self.delta_n - alpha_p.detach().mean())

        pos_term = -self.gamma * alpha_p * (pos_similarities - self.delta_p)
        log_pos = torch.logsumexp(pos_term, dim=1, keepdim=True)

        neg_term = self.gamma * alpha_n * (neg_similarities - self.delta_n)
        log_neg = torch.logsumexp(neg_term, dim=1, keepdim=True)

        loss_per_sample = nn.Softplus()(log_neg + log_pos)

        if self.reduction == 'mean':
            return loss_per_sample.mean()
        elif self.reduction == 'sum':
            return loss_per_sample.sum()
        else:
            return loss_per_sample


class CAAHL(nn.Module):
    """Cohesion-Aware Adaptive Hybrid Loss (CAAHL).

    Dynamically adjusts the weighting between InfoNCE (contrastive) and Circle
    (margin-based) losses according to the batch-wise intra-modal feature
    cohesion, which is estimated as the proportion of cross-location sample
    pairs whose similarity exceeds a threshold.
    """

    def __init__(self,
                 loss_function,
                 base_temperature=0.07,
                 circle_margin=0.25,
                 circle_gamma=256,
                 initial_circle_weight=0.3,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.loss_function = loss_function
        self.base_temperature = base_temperature
        self.circle_margin = circle_margin
        self.circle_gamma = circle_gamma
        self.device = device

        self.initial_circle_weight = initial_circle_weight
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, drone_features, sat_features, logit_scale):
        if self.current_epoch == -1:
            # InfoNCE only
            info_nce_loss_d2s = self.info_nce_loss(drone_features, sat_features, logit_scale)
            info_nce_loss_s2d = self.info_nce_loss(sat_features, drone_features, logit_scale)
            return (info_nce_loss_d2s + info_nce_loss_s2d) / 2

        labels = torch.arange(len(drone_features), dtype=torch.long, device=self.device)

        cohesion_drone = self.compute_cohesion(drone_features)
        cohesion_sat = self.compute_cohesion(sat_features)

        info_nce_weight, circle_weight = self.compute_loss_weights(
            (cohesion_drone + cohesion_sat) / 2
        )

        info_nce_loss_d2s = self.info_nce_loss(drone_features, sat_features, logit_scale)
        info_nce_loss_s2d = self.info_nce_loss(sat_features, drone_features, logit_scale)
        info_nce_loss = (info_nce_loss_d2s + info_nce_loss_s2d) / 2

        circle_loss_d2s = self.circle_loss(drone_features, sat_features)
        circle_loss_s2d = self.circle_loss(sat_features, drone_features)
        circle_loss = (circle_loss_d2s + circle_loss_s2d) / 2

        total_loss = info_nce_weight * info_nce_loss + circle_weight * circle_loss
        return total_loss

    def compute_cohesion(self, features):
        """Proportion of cross-location pairs with similarity above a threshold."""
        features = F.normalize(features, dim=-1)
        sim_matrix = features @ features.T

        neg_mask = ~torch.eye(features.size(0), dtype=torch.bool, device=self.device)
        neg_sim = sim_matrix[neg_mask]

        high_sim_ratio = (neg_sim > 0.75).float().mean()
        return high_sim_ratio.item()

    def compute_loss_weights(self, cohesion):
        base_circle_weight = self.initial_circle_weight
        noise_adjustment = 1.0 - cohesion
        adaptive_circle_weight = max(0, base_circle_weight * noise_adjustment)

        circle_weight = adaptive_circle_weight
        info_nce_weight = 1.0 - circle_weight
        return info_nce_weight, circle_weight

    def info_nce_loss(self, query, key, logit_scale):
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)

        logits = logit_scale * query @ key.T
        labels = torch.arange(len(logits), dtype=torch.long, device=self.device)

        return self.loss_function(logits, labels)

    def circle_loss(self, query, key):
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)

        sim_matrix = query @ key.T
        batch_size = sim_matrix.shape[0]

        pos_mask = torch.eye(batch_size, dtype=torch.bool, device=self.device)
        neg_mask = ~pos_mask

        pos_similarities = sim_matrix[pos_mask].unsqueeze(1)
        neg_similarities = sim_matrix[neg_mask].view(batch_size, batch_size - 1)

        alpha_p = torch.relu(1 - self.circle_margin - pos_similarities.detach())
        alpha_n = torch.relu(neg_similarities.detach() - self.circle_margin)

        pos_term = -self.circle_gamma * alpha_p * (pos_similarities - 1 + self.circle_margin)
        log_pos = torch.logsumexp(pos_term, dim=1, keepdim=True)

        neg_term = self.circle_gamma * alpha_n * (neg_similarities - self.circle_margin)
        log_neg = torch.logsumexp(neg_term, dim=1, keepdim=True)

        loss_per_sample = nn.Softplus()(log_neg + log_pos)
        return loss_per_sample.mean()
