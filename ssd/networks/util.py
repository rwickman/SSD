import torch
import torch.nn as nn

import numpy as np

from ssd.networks.spectral_norm import spectral_norm


def create_mlp(
        inp_size,
        out_size,
        n_layers,
        hidden_size,
        use_spectral_norm=False,
        spectral_ceof=1.0,
        hidden_w_init=nn.init.xavier_normal_,
        hidden_b_init=nn.init.zeros_,
        bias=True):

    layers = []
    
    # Add the hidden layers
    cur_inp_size = inp_size
    for i in range(n_layers):
        if use_spectral_norm:
            linear_layer = spectral_norm(
                    nn.Linear(cur_inp_size, hidden_size, bias=bias), spectral_coef=spectral_ceof) 
        else:
            linear_layer = nn.Linear(cur_inp_size, hidden_size, bias=bias)

        # Initialize the weights
        hidden_w_init(linear_layer.weight)
        if bias:
            hidden_b_init(linear_layer.bias)

        layers.append(linear_layer)
        layers.append(nn.ReLU(inplace=True))
        cur_inp_size = hidden_size

    # Add the output layer    
    if use_spectral_norm:
        out_layer = spectral_norm(
                nn.Linear(hidden_size, out_size), spectral_coef=spectral_ceof)
    else:
        out_layer = nn.Linear(hidden_size, out_size, bias=bias)

    # Initialize the weights
    hidden_w_init(out_layer.weight)
    if bias:
        hidden_b_init(out_layer.bias)

    layers.append(out_layer)

    return nn.Sequential(*layers)


def compute_knn_weights(x, k):
    with torch.no_grad():
        sorted_dist = knn(x, k)
        sqr_dist = (sorted_dist ** 2).mean(dim=1)
        weights = sqr_dist / (sqr_dist.sum() + 1e-5)

    return weights


def knn(x, k):
    # Take the difference between every pair
    diff = x[:, None] - x[None:, ]
    # Compute the distance
    dist = (diff ** 2).sum(dim=-1) ** 0.5
    # Sort the distances
    sorted_dist, _ = dist.sort()
    # Compute the average distance, excluding itself
    sorted_dist = sorted_dist[:, 1:k+1]

    return sorted_dist


class RunningMeanStd(nn.Module):
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        super().__init__()
        # self.mean = 
        # self.var = torch.ones(shape)
        # self.count = torch.tensor(epsilon)

        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(epsilon))
        

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )
    
    def normalize(self, x):
        return (x - self.mean) / (self.var ** 0.5)


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count
