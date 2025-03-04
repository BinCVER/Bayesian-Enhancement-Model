import torch
import torch.nn as nn
import torch.distributions as distributions
from itertools import repeat
import collections
from abc import abstractmethod

class BaseLayer_(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        if not self.deterministic:
            return self._forward_uncertain(input)
        else:
            return self._forward_det(input)

    @abstractmethod
    def _forward_uncertain(self, input):
        pass

    @abstractmethod
    def _forward_det(self, input):
        pass

    def kl_div(self, mu_q, sigma_q, mu_p, sigma_p):
        """
        Calculates kl divergence between two gaussians (Q || P)

        Parameters:
             * mu_q: torch.Tensor -> mu parameter of distribution Q
             * sigma_q: torch.Tensor -> sigma parameter of distribution Q
             * mu_p: float -> mu parameter of distribution P
             * sigma_p: float -> sigma parameter of distribution P

        returns torch.Tensor of shape 0
        """
        kl = torch.log(sigma_p) - torch.log(sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 * (sigma_p**2)) - 0.5
        return kl.mean()
