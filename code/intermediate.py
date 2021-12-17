from math import ceil

import numpy as np
from scipy.stats import norm, binom_test
from statsmodels.stats.proportion import proportion_confint
import torch
import torch.nn as nn

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

    def forward(self, x):
        x = torch.clip(x, 0., 1.)
        attacker = PGD_L2(steps=self.num_steps, device='cuda', max_norm=self.epsilon)
        high_confidence_image = attacker.attack(self.base_classifier, x, None, entropy_attack=True, entropy_samples=self.entropy_samples)
        return self.base_classifier(high_confidence_image)
