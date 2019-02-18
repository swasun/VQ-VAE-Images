from encoder import Encoder
from decoder import Decoder
from vector_quantizer import VectorQuantizer
from vector_quantizer_ema import VectorQuantizerEMA

import torch.nn as nn
import torch
import os


class AutoEncoder(nn.Module):
    
    def __init__(self, device, configuration):
        super(AutoEncoder, self).__init__()
        
        self._encoder = Encoder(
            3,
            configuration.num_hiddens,
            configuration.num_residual_layers, 
            configuration.num_residual_hiddens
        )

        self._pre_vq_conv = nn.Conv2d(
            in_channels=configuration.num_hiddens, 
            out_channels=configuration.embedding_dim,
            kernel_size=1, 
            stride=1
        )

        if configuration.decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(
                device,
                configuration.num_embeddings,
                configuration.embedding_dim, 
                configuration.commitment_cost,
                configuration.decay
            )
        else:
            self._vq_vae = VectorQuantizer(
                device,
                configuration.num_embeddings,
                configuration.embedding_dim,
                configuration.commitment_cost
            )

        self._decoder = Decoder(
            configuration.embedding_dim,
            configuration.num_hiddens, 
            configuration.num_residual_layers, 
            configuration.num_residual_hiddens
        )

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity

    def save(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(self, path, configuration, device):
        model = AutoEncoder(device, configuration)
        model.load_state_dict(torch.load(path, map_location=device))
        return model

    @property
    def vq_vae(self):
        return self._vq_vae
