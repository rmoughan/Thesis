from infrastructure.utils import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

class ConvNet(nn.Module):

    def __init__(self, config):
        super(ConvNet, self).__init__()
        input_h, input_w, input_c = config.params.get("input_h"), config.params.get("input_w"), config.params.get("input_c")
        output_size = config.params["output_size"]
        self.num_conv_layers = len(config.params.get("num_channels", []))
        num_channels = config.params.get("num_channels")
        kernel_sizes, stride_sizes = config.params.get("kernel_sizes"), config.params.get("stride_sizes")
        self.num_linear_layers = len(config.params.get("linear_layer_dims", []))
        self.linear_layer_dims = config.params.get("linear_layer_dims")
        self.linear_input_dim = config.params.get("linear_input_dim", 0)
        self.activation = config.params.get("activation_fn", "ReLU")
        self.activation = eval("F." + self.activation)

        self.conv_layers = nn.ModuleList([])
        for layer in range(self.num_conv_layers):
            new_layer = nn.Conv2d(in_channels = input_c, out_channels = num_channels[layer],
                    kernel_size = kernel_sizes[layer], stride = stride_sizes[layer])
            torch.nn.init.xavier_uniform_(new_layer.weight)
            torch.nn.init.constant_(new_layer.bias.data, 0)
            self.conv_layers.append(new_layer)
            input_c = num_channels[layer]

        conv_out_h, conv_out_w = input_h, input_w

        for layer in range(self.num_conv_layers):
            conv_out_h = conv2d_output_dimension(conv_out_h,
                        kernel_size = kernel_sizes[layer], stride = stride_sizes[layer])
            conv_out_w = conv2d_output_dimension(conv_out_w,
                        kernel_size = kernel_sizes[layer], stride = stride_sizes[layer])
        conv_out_dim = num_channels[-1] * conv_out_h * conv_out_w if self.num_conv_layers > 0 else 0

        self.linear_layers = nn.ModuleList([])
        in_dim = conv_out_dim if self.num_conv_layers != 0 else self.linear_input_dim
        for layer in range(self.num_linear_layers):
            new_layer = nn.Linear(in_dim, self.linear_layer_dims[layer])
            torch.nn.init.xavier_uniform_(new_layer.weight.data)
            torch.nn.init.constant_(new_layer.bias.data, 0)
            self.linear_layers.append(new_layer)
            in_dim = self.linear_layer_dims[layer]

        self.head = nn.Linear(self.linear_layer_dims[-1], output_size)
        torch.nn.init.xavier_uniform_(self.head.weight.data)
        torch.nn.init.constant_(self.head.bias.data, 0)

    def forward(self, x):
        for layer in self.conv_layers:
            x = self.activation(layer(x))
        # x = x.permute(0,2,3,1)
        x = x.flatten(start_dim = 1) if self.num_conv_layers != 0 else x
        for layer in self.linear_layers:
            x = self.activation(layer(x))
        x = x.unsqueeze(dim = 0) if len(x.shape) == 1 else x
        x = x.view(x.size(0), -1)
        out = self.head(x)
        return out
