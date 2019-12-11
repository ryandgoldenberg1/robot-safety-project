from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Transition = namedtuple('Transition', ('obs', 'action', 'reward', 'next_obs', 'done'))


def stringify(instance):
    name = instance.__class__.__name__
    result = name + '(\n'
    for k, v in instance.__dict__.items():
        v = str(v)
        result += f'  {k}: {v}\n'
    result += ')'
    return result


def create_mlp(layer_sizes):
    layers = []
    for i in range(len(layer_sizes) - 1):
        input_size = layer_sizes[i]
        output_size = layer_sizes[i + 1]
        layers.append(nn.Linear(input_size, output_size))
        layers.append(nn.Tanh())
    layers.pop()
    return nn.Sequential(*layers)


def create_conv_net(obs_shape, kernel_sizes, paddings, channel_sizes, fc_sizes):
    assert len(obs_shape) == 3
    num_conv_layers = len(kernel_sizes)
    assert len(paddings) == num_conv_layers
    assert len(channel_sizes) == num_conv_layers
    in_channels, width, height = obs_shape
    layers = []
    for i in range(num_conv_layers):
        out_channels = channel_sizes[i]
        kernel_size = kernel_sizes[i]
        padding = paddings[i]
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                padding=padding))
        layers.append(nn.ReLU())
        in_channels = out_channels
        width += -(kernel_size - 1) + 2 * padding
        height += -(kernel_size - 1) + 2 * padding
    layers.append(nn.Flatten())
    input_size = width * height * in_channels
    for output_size in fc_sizes:
        layers.append(nn.Linear(input_size, output_size))
        layers.append(nn.ReLU())
        input_size = output_size
    layers.pop()
    return nn.Sequential(*layers)


class GaussianMlp(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_sizes, max_action, min_variance, max_variance):
        super().__init__()
        assert len(obs_shape) == 1, f'unsupported obs_shape: {obs_shape}'
        assert len(action_shape) == 1, f'unsupported action_shape: {action_shape}'
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.max_action = max_action
        self.min_variance = min_variance
        self.max_variance = max_variance

        body_layer_sizes = [obs_shape[0]] + hidden_sizes
        last_hidden_size = body_layer_sizes[-1]
        action_size = action_shape[0]
        self.body = create_mlp(body_layer_sizes)
        self.mean_head = nn.Linear(last_hidden_size, action_size)
        self.var_head = nn.Linear(last_hidden_size, action_size)

    def forward(self, x):
        bs = x.shape[0]
        assert x.shape == (bs, *self.obs_shape)
        x = torch.tanh(self.body(x))
        mean = torch.tanh(self.mean_head(x)) * self.max_action
        var = F.softplus(self.var_head(x))
        var = torch.clamp(var, min=self.min_variance, max=self.max_variance)
        return mean, var


class CategoricalProjection(nn.Module):
    def __init__(self, v_min, v_max, num_atoms, discount_factor):
        super().__init__()
        assert v_max > v_min
        assert num_atoms > 1
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms
        self.discount_factor = discount_factor
        self.atom_delta = (v_max - v_min) / (num_atoms - 1)
        self.atom_values = torch.FloatTensor([v_min + self.atom_delta * i for i in range(self.num_atoms)])

    def forward(self, reward, probs, not_done):
        bs = probs.shape[0]
        assert reward.shape == (bs, 1)
        assert not_done.shape == (bs, 1)
        assert probs.shape == (bs, self.num_atoms)

        atom_values = self.atom_values.unsqueeze(0).expand((bs, self.num_atoms))
        reward = reward.expand((bs, self.num_atoms))
        new_atom_values = reward + self.discount_factor * not_done * atom_values
        new_atom_values = torch.clamp(new_atom_values, min=self.v_min, max=self.v_max)

        new_atom_indices = (new_atom_values - self.v_min) / self.atom_delta
        lower_index = torch.floor(new_atom_indices)
        upper_index = torch.ceil(new_atom_indices)
        lower_coef = upper_index - new_atom_indices
        upper_coef = new_atom_indices - lower_index

        projection_matrix = torch.zeros((bs, self.num_atoms, self.num_atoms))
        for i in range(bs):
            for j in range(self.num_atoms):
                l = int(lower_index[i, j].item())
                l_coef = lower_coef[i, j]
                u = int(upper_index[i, j].item())
                u_coef = upper_coef[i, j]
                projection_matrix[i, l, j] = l_coef
                projection_matrix[i, u, j] = u_coef
                if l == u:
                    assert l_coef == 0
                    assert u_coef == 0
                    projection_matrix[i, l, j] = 1.

        probs = probs.unsqueeze(-1)
        assert probs.shape == (bs, self.num_atoms, 1)
        new_probs = torch.bmm(projection_matrix, probs).squeeze(-1)
        assert new_probs.shape == (bs, self.num_atoms)
        return new_probs
