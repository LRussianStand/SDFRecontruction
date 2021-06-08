#-*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F


class TransSDF(nn.Module):
    """暂且只输出d"""
    def __init__(
        self,
        input_size,
        output_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        weight_norm=False,
        norm_layers=(),
        res_in=(),
        use_tanh=False,
        input_dropout=False,
    ):
        super(TransSDF, self).__init__()

        def make_sequence():
            return []

        self.input_dropout = input_dropout
        dims = [input_size] + dims + [output_size]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers

        self.weight_norm = weight_norm
        self.res_in = res_in
        for layer in range(0, self.num_layers - 1):
            if layer in res_in:
                in_dim = dims[layer] + dims[0]
            else:
                in_dim = dims[layer]

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(in_dim, dims[layer+1])),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(in_dim, dims[layer+1]))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(dims[layer+1]))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input):
        xyz = input[:, -3:]

        if self.input_dropout:
            x = F.dropout(input,0.2)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.res_in:
                x = torch.cat([x, input], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x