from residual_stack import ResidualStack

import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens,
            kernel_size=3, 
            stride=1,
            padding=1
        )
        
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens
        )
        
        self._conv_trans_1 = nn.ConvTranspose2d(
            in_channels=num_hiddens, 
            out_channels=num_hiddens//2,
            kernel_size=4, 
            stride=2,
            padding=1
        )
        
        self._conv_trans_2 = nn.ConvTranspose2d(
            in_channels=num_hiddens//2, 
            out_channels=3,
            kernel_size=4, 
            stride=2,
            padding=1
        )

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        return self._conv_trans_2(x)
