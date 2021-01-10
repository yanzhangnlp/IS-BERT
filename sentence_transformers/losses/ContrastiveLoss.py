from enum import Enum
from typing import Iterable, Dict

import torch.nn.functional as F
from torch import nn, Tensor

from sentence_transformers.SentenceTransformer import SentenceTransformer

class SiameseDistanceMetric(Enum):
    """
    The metric for the contrastive loss
    """
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1-F.cosine_similarity(x, y)



class ContrastiveLoss(nn.Module):

    def __init__(self, model: SentenceTransformer, temperature=0.07, base_temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.model = model
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        contrast_count = len(reps)
        assert contrast_count == 2
        rep_anchor, rep_other = reps
        batch_size = rep_anchor.size(0)
        features = torch.cat([rep_anchor.view(batch_size, 1, -1), rep_other.view(batch_size, 1, -1)], dim=1)
        contrast_feature = torch.cat(reps, dim=0)
        anchor_feature = contrast_feature
        mask = torch.eye(batch_size, dtype=torch.float32)
        anchor_count = contrast_count
        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1), 0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss











