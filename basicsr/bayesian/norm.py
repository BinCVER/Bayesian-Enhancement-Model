import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from itertools import repeat
from torch.nn import Parameter
from .base_layer import BaseLayer_
import math

class LayerNorm2dReparameterization(BaseLayer_):
    def __init__(self,
                 normalized_shape,
                 eps=1e-05,
                 bias=True,
                 sigma_init=0.05,
                 decay=0.9998):

        super(LayerNorm2dReparameterization, self).__init__()

        self.deterministic = False # set to True to get deterministic output
        self.bias = bias
        self.decay = decay
        self.sigma_init = sigma_init
        self.normalized_shape = normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.step = 0


        self.mu_weight = Parameter(torch.Tensor(normalized_shape))
        self.rho_weight = Parameter(torch.Tensor(normalized_shape))
        self.register_buffer('eps_weight', torch.Tensor(normalized_shape), persistent=False)
        self.register_buffer('prior_mu_weight', torch.Tensor(normalized_shape), persistent=False)
        self.register_buffer('prior_rho_weight', torch.Tensor(normalized_shape), persistent=False)
        if bias:
            self.mu_bias = Parameter(torch.Tensor(normalized_shape))
            self.rho_bias = Parameter(torch.Tensor(normalized_shape))
            self.register_buffer('eps_bias', torch.Tensor(normalized_shape), persistent=False)
            self.register_buffer('prior_mu_bias', torch.Tensor(normalized_shape), persistent=False)
            self.register_buffer('prior_rho_bias', torch.Tensor(normalized_shape), persistent=False)


        self.init_parameters()

    def init_parameters(self):
        rho_init = math.log(math.expm1(abs(self.sigma_init)) + 1e-20)
        self.mu_weight.data.fill_(1)
        self.rho_weight.data.fill_(rho_init)

        self.prior_mu_weight.data.copy_(self.mu_weight.data)
        self.prior_rho_weight.data.copy_(self.rho_weight.data)

        if self.bias:
            self.mu_bias.data.fill_(0)
            self.rho_bias.data.fill_(rho_init)

            self.prior_mu_bias.data.copy_(self.mu_bias.data)
            self.prior_rho_bias.data.copy_(self.rho_bias.data)

    def kl_loss(self):
        kl = self.kl_div(self.mu_weight, self.sigma_weight, self.prior_mu_weight, self.prior_sigma_weight)
        if self.bias:
            kl += self.kl_div(self.mu_bias, self.sigma_bias, self.prior_mu_bias, self.prior_sigma_bias)
        return kl

    def _forward_uncertain(self, x):
        if self.training:
            with torch.no_grad():
                _decay = min(self.decay, (1 + self.step) / (10 + self.step))
                # _decay = self.decay
                self.prior_mu_weight = _decay * self.prior_mu_weight + (1 - _decay) * self.mu_weight
                self.prior_rho_weight = _decay * self.prior_rho_weight + (1 - _decay) * self.rho_weight
                self.prior_sigma_weight = torch.log1p(torch.exp(self.prior_rho_weight))

                if self.bias:
                    self.prior_mu_bias = _decay * self.prior_mu_bias + (1 - _decay) * self.mu_bias
                    self.prior_rho_bias = _decay * self.prior_rho_bias + (1 - _decay) * self.rho_bias
                    self.prior_sigma_bias = torch.log1p(torch.exp(self.prior_rho_bias))
            self.step += 1

        self.sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        weight = self.mu_weight + self.sigma_weight * self.eps_weight.data.normal_()

        if self.bias:
            self.sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            bias = self.mu_bias + self.sigma_bias * self.eps_bias.data.normal_()
        else:
            bias = None

        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
        x = x.permute(0, 3, 1, 2)

        return x

    def _forward_det(self, x):

        weight = self.mu_weight
        if self.bias:
            bias = self.mu_bias
        else:
            bias = None

        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
        x = x.permute(0, 3, 1, 2)

        return x


class InstanceNorm2dReparameterization(BaseLayer_):
    def __init__(self, num_features,
                 eps=1e-5,
                 momentum=0.1,
                 bias=True,
                 affine=True,
                 track_running_stats=True,
                 sigma_init=0.05,
                 decay=0.9998):
        """
        Args:
            num_features (int): Number of channels in the input.
            eps (float): Small value added to variance for numerical stability.
            momentum (float): Value for updating running mean and variance.
            affine (bool): If True, learnable weight and bias are added.
            track_running_stats (bool): If True, running mean and variance are tracked.
        """
        super(InstanceNorm2dReparameterization, self).__init__()
        self.deterministic = False # set to True to get deterministic output
        self.decay = decay
        self.sigma_init = sigma_init

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.bias = bias
        self.step = 0

        if not affine:
            raise ValueError('Arg affine must be True')
        if not bias:
            raise ValueError('Arg bias must be True')

        self.mu_weight = Parameter(torch.Tensor(num_features))
        self.rho_weight = Parameter(torch.Tensor(num_features))
        self.register_buffer('eps_weight', torch.Tensor(num_features), persistent=False)
        self.register_buffer('prior_mu_weight', torch.Tensor(num_features), persistent=False)
        self.register_buffer('prior_rho_weight', torch.Tensor(num_features), persistent=False)

        if bias:
            self.mu_bias = Parameter(torch.Tensor(num_features))
            self.rho_bias = Parameter(torch.Tensor(num_features))
            self.register_buffer('eps_bias', torch.Tensor(num_features), persistent=False)
            self.register_buffer('prior_mu_bias', torch.Tensor(num_features), persistent=False)
            self.register_buffer('prior_rho_bias', torch.Tensor(num_features), persistent=False)

        self.init_parameters()


        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.running_mean = None
            self.running_var = None

    def init_parameters(self):
        rho_init = math.log(math.expm1(abs(self.sigma_init)) + 1e-20)
        self.mu_weight.data.fill_(1)
        self.rho_weight.data.fill_(rho_init)

        self.prior_mu_weight.data.copy_(self.mu_weight.data)
        self.prior_rho_weight.data.copy_(self.rho_weight.data)

        if self.bias:
            self.mu_bias.data.fill_(0)
            self.rho_bias.data.fill_(rho_init)

            self.prior_mu_bias.data.copy_(self.mu_bias.data)
            self.prior_rho_bias.data.copy_(self.rho_bias.data)


    def kl_loss(self):
        kl = self.kl_div(self.mu_weight, self.sigma_weight, self.prior_mu_weight, self.prior_sigma_weight)
        if self.bias:
            kl += self.kl_div(self.mu_bias, self.sigma_bias, self.prior_mu_bias, self.prior_sigma_bias)
        return kl

    def _forward_uncertain(self, x):
        if self.training:
            with torch.no_grad():
                _decay = min(self.decay, (1 + self.step) / (10 + self.step))
                self.prior_mu_weight = _decay * self.prior_mu_weight + (1 - _decay) * self.mu_weight
                self.prior_rho_weight = _decay * self.prior_rho_weight + (1 - _decay) * self.rho_weight
                self.prior_sigma_weight = torch.log1p(torch.exp(self.prior_rho_weight))

                if self.bias:
                    self.prior_mu_bias = _decay * self.prior_mu_bias + (1 - _decay) * self.mu_bias
                    self.prior_rho_bias = _decay * self.prior_rho_bias + (1 - _decay) * self.rho_bias
                    self.prior_sigma_bias = torch.log1p(torch.exp(self.prior_rho_bias))
            self.step += 1
            if self.track_running_stats:
                self.num_batches_tracked += 1

        self.sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        weight = self.mu_weight + self.sigma_weight * self.eps_weight.data.normal_()

        if self.bias:
            self.sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            bias = self.mu_bias + self.sigma_bias * self.eps_bias.data.normal_()
        else:
            bias = None

        return F.instance_norm(x, running_mean=self.running_mean,
                            running_var=self.running_var, weight=weight, bias=bias,
                            use_input_stats=not self.track_running_stats, momentum=self.momentum, eps=self.eps)
        # # Training mode: use input stats
        # if self.training or not self.track_running_stats:
        #     mean = input.mean(dim=(2, 3), keepdim=True)
        #     var = input.var(dim=(2, 3), keepdim=True, unbiased=False)
        #     if self.track_running_stats:
        #         with torch.no_grad():
        #             self.running_mean = (
        #                 self.momentum * mean.mean(dim=0).squeeze() + (1 - self.momentum) * self.running_mean
        #             )
        #             self.running_var = (
        #                 self.momentum * var.mean(dim=0).squeeze() + (1 - self.momentum) * self.running_var
        #             )
        # else:
        #     # Evaluation mode: use running stats
        #     mean = self.running_mean.view(1, -1, 1, 1)
        #     var = self.running_var.view(1, -1, 1, 1)

        # # Normalize input
        # input_normalized = (input - mean) / torch.sqrt(var + self.eps)

        # self.sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        # weight = self.mu_weight + self.sigma_weight * self.eps_weight.data.normal_()

        # if self.bias:
        #     self.sigma_bias = torch.log1p(torch.exp(self.rho_bias))
        #     bias = self.mu_bias + self.sigma_bias * self.eps_bias.data.normal_()
        # else:
        #     bias = None

        # # Apply affine transformation if enabled
        # weight = weight.view(1, -1, 1, 1)
        # input_normalized = input_normalized * weight
        # if self.bias:
        #     bias = bias.view(1, -1, 1, 1)
        #     input_normalized = input_normalized  + bias

        # return input_normalized


    def _forward_det(self, x):


        weight = self.mu_weight
        if self.bias:
            bias = self.mu_bias
        else:
            bias = None

        return F.instance_norm(x, running_mean=self.running_mean,
                            running_var=self.running_var, weight=weight, bias=bias,
                            use_input_stats=not self.track_running_stats, momentum=self.momentum, eps=self.eps)



        # # use input stats
        # if not self.track_running_stats:
        #     mean = input.mean(dim=(2, 3), keepdim=True)
        #     var = input.var(dim=(2, 3), keepdim=True, unbiased=False)
        # else:
        #     # use running stats
        #     mean = self.running_mean.view(1, -1, 1, 1)
        #     var = self.running_var.view(1, -1, 1, 1)

        # # Normalize input
        # input_normalized = (input - mean) / torch.sqrt(var + self.eps)

        # self.sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        # weight = self.mu_weight + self.sigma_weight * self.eps_weight.data.normal_()
        # # Apply affine transformation if enabled
        # weight = weight.view(1, -1, 1, 1)
        # input_normalized = input_normalized * weight
        # if self.bias:
        #     bias = bias.view(1, -1, 1, 1)
        #     input_normalized = input_normalized  + bias

        # return input_normalized
