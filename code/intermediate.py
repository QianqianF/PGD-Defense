from math import ceil

import numpy as np
from scipy.stats import norm, binom_test
from statsmodels.stats.proportion import proportion_confint
import torch
import torch.nn as nn
import math

from attacks import PGD_L2

class Intermediate(nn.Module):
    """An PGD based intermediate layer to attack the uncertainty before doing a classification"""

    def __init__(self, base_classifier: torch.nn.Module, num_steps: int, epsilon: float, entropy_samples: int = 1):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        super().__init__()
        self.base_classifier = base_classifier
        self.num_steps = num_steps
        self.epsilon = epsilon
        self.entropy_samples = entropy_samples

    def forward(self, batch, noise):
        print('initial stddev:', math.sqrt(torch.mean(noise * noise).cpu().item()))
        x = batch + noise
        x = torch.clip(x, 0., 1.)
        attacker = PGD_L2(steps=1, device='cuda', max_norm=self.epsilon / self.num_steps)
        for i in range(self.num_steps):
            x = attacker.attack(self.base_classifier, x, None, entropy_attack=True, entropy_samples=self.entropy_samples)
            print('stddev after PGD step ' + str(i) + ':', math.sqrt(torch.mean((x - batch) * (x - batch)).cpu().item()))
        return self.base_classifier(x)
