import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
import torch.nn.functional as F
from ..SentenceTransformer import SentenceTransformer
import torch
import torch.nn as nn
import numpy as np
import logging
import math

class MutualInformationLoss(nn.Module):
    def __init__(self,
                 model: SentenceTransformer,
                 sentence_embedding_dimension: int):
        super(MutualInformationLoss, self).__init__()
        self.model = model

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):

        rep = [self.model(sentence_feature) for sentence_feature in sentence_features][0]
        tok_rep = rep['token_embeddings']

        sentence_lengths = torch.clamp(rep['sentence_lengths'], min=1).data.cpu().numpy() 
        tok_rep = [tok_rep[i][:sentence_lengths[i]] for i in range(len(sentence_lengths))]
        local_rep = torch.cat(tok_rep, dim=0)

        global_rep = rep['sentence_embedding']
        pos_mask, neg_mask = self.create_masks(sentence_lengths)

        mode='fd'
        measure='JSD'
        local_global_loss = self.local_global_loss_(local_rep, global_rep, pos_mask, neg_mask, measure)

        return local_global_loss


        # reps = self.model(sentence_features)
        # tok_rep = reps['token_embeddings'] 
        # sentence_lengths = torch.clamp(reps['sentence_lengths'], min=1).data.cpu().numpy() 

        # tok_rep = [tok_rep[i][:sentence_lengths[i]] for i in range(len(sentence_lengths))]
        # local_rep = torch.cat(tok_rep, dim=0)

        # global_rep = reps['sentence_embedding'] 

        # pos_mask, neg_mask = self.create_masks(sentence_lengths)

        # mode='fd'
        # measure='JSD'
        # local_global_loss = self.local_global_loss_(local_rep, global_rep, pos_mask, neg_mask, measure)

        # return local_global_loss


    def create_masks(self, lens_a):
        pos_mask = torch.zeros((np.sum(lens_a), len(lens_a))).cuda()
        neg_mask = torch.ones((np.sum(lens_a), len(lens_a))).cuda()
        temp = 0
        for idx in range(len(lens_a)):
            for j in range(temp, lens_a[idx] + temp):
                pos_mask[j][idx] = 1.
                neg_mask[j][idx] = 0.
            temp += lens_a[idx]

        return pos_mask, neg_mask


    def local_global_loss_(self, l_enc, g_enc, pos_mask, neg_mask, measure):
        '''
        Args:
            l: Local feature map.
            g: Global features.
            measure: Type of f-divergence. For use with mode `fd`
            mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
        Returns:
            torch.Tensor: Loss.
        '''

        res = torch.mm(l_enc, g_enc.t())

        # print(l_enc.size(), res.size(), pos_mask.size())
        num_nodes = pos_mask.size(0)
        num_graphs = pos_mask.size(1)
        E_pos = self.get_positive_expectation(res * pos_mask, measure, average=False).sum()
        E_pos = E_pos / num_nodes
        E_neg = self.get_negative_expectation(res * neg_mask, measure, average=False).sum()
        E_neg = E_neg / (num_nodes * (num_graphs - 1))

        return E_neg - E_pos



    def log_sum_exp(self, x, axis=None):
        """Log sum exp function

        Args:
            x: Input.
            axis: Axis over which to perform sum.

        Returns:
            torch.Tensor: log sum exp

        """
        x_max = torch.max(x, axis)[0]
        y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
        return y



    def raise_measure_error(self, measure):
        supported_measures = ['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1']
        raise NotImplementedError(
            'Measure `{}` not supported. Supported: {}'.format(measure,
                                                               supported_measures))


    def get_positive_expectation(self, p_samples, measure, average=True):
        """Computes the positive part of a divergence / difference.

        Args:
            p_samples: Positive samples.
            measure: Measure to compute for.
            average: Average the result over samples.

        Returns:
            torch.Tensor

        """
        log_2 = math.log(2.)

        if measure == 'GAN':
            Ep = - F.softplus(-p_samples)
        elif measure == 'JSD':
            Ep = log_2 - F.softplus(- p_samples)
        elif measure == 'X2':
            Ep = p_samples ** 2
        elif measure == 'KL':
            Ep = p_samples + 1.
        elif measure == 'RKL':
            Ep = -torch.exp(-p_samples)
        elif measure == 'DV':
            Ep = p_samples
        elif measure == 'H2':
            Ep = 1. - torch.exp(-p_samples)
        elif measure == 'W1':
            Ep = p_samples
        else:
            raise_measure_error(measure)

        if average:
            return Ep.mean()
        else:
            return Ep


    def get_negative_expectation(self, q_samples, measure, average=True):
        """Computes the negative part of a divergence / difference.

        Args:
            q_samples: Negative samples.
            measure: Measure to compute for.
            average: Average the result over samples.

        Returns:
            torch.Tensor

        """
        log_2 = math.log(2.)

        if measure == 'GAN':
            Eq = F.softplus(-q_samples) + q_samples
        elif measure == 'JSD':
            Eq = F.softplus(-q_samples) + q_samples - log_2
        elif measure == 'X2':
            Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
        elif measure == 'KL':
            Eq = torch.exp(q_samples)
        elif measure == 'RKL':
            Eq = q_samples - 1.
        elif measure == 'DV':
            Eq = self.log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
        elif measure == 'H2':
            Eq = torch.exp(q_samples) - 1.
        elif measure == 'W1':
            Eq = q_samples
        else:
            self.raise_measure_error(measure)

        if average:
            return Eq.mean()
        else:
            return Eq
