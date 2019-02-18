 #####################################################################################
 # MIT License                                                                       #
 #                                                                                   #
 # Copyright (C) 2019 Charly Lamothe                                                 #
 # Copyright (C) 2018 Zalando Research                                               #
 #                                                                                   #
 # This file is part of VQ-VAE-images.                                               #
 #                                                                                   #
 #   Permission is hereby granted, free of charge, to any person obtaining a copy    #
 #   of this software and associated documentation files (the "Software"), to deal   #
 #   in the Software without restriction, including without limitation the rights    #
 #   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell       #
 #   copies of the Software, and to permit persons to whom the Software is           #
 #   furnished to do so, subject to the following conditions:                        #
 #                                                                                   #
 #   The above copyright notice and this permission notice shall be included in all  #
 #   copies or substantial portions of the Software.                                 #
 #                                                                                   #
 #   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR      #
 #   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        #
 #   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE     #
 #   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER          #
 #   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   #
 #   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE   #
 #   SOFTWARE.                                                                       #
 #####################################################################################

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os


class Trainer(object):

    def __init__(self, device, model, optimizer, dataset, verbose=True):
        self._device = device
        self._model = model
        self._optimizer = optimizer
        self._dataset = dataset
        self._verbose = verbose
        self._train_res_recon_error = []
        self._train_res_perplexity = []

    def train(self, num_training_updates):
        self._model.train()

        for i in range(num_training_updates):
            (data, _) = next(iter(self._dataset.training_loader))
            data = data.to(self._device)
            self._optimizer.zero_grad()

            """
            The perplexity a useful value to track during training.
            It indicates how many codes are 'active' on average.
            """
            vq_loss, data_recon, perplexity = self._model(data)
            recon_error = torch.mean((data_recon - data)**2) / self._dataset.train_data_variance
            loss = recon_error + vq_loss
            loss.backward()

            self._optimizer.step()
            
            self._train_res_recon_error.append(recon_error.item())
            self._train_res_perplexity.append(perplexity.item())

            if self._verbose and (i % (num_training_updates / 10) == 0):
                print('%d iterations' % (i+1))
                print('reconstruction error: %.3f' % np.mean(self._train_res_recon_error[-100:]))
                print('perplexity: %.3f' % np.mean(self._train_res_perplexity[-100:]))
                print()

    def save_loss_plot(self, path):
        train_res_recon_error_smooth = savgol_filter(self._train_res_recon_error, 201, 7)
        train_res_perplexity_smooth = savgol_filter(self._train_res_perplexity, 201, 7)
        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot(1,2,1)
        ax.plot(train_res_recon_error_smooth)
        ax.set_yscale('log')
        ax.set_title('Smoothed NMSE.')
        ax.set_xlabel('iteration')

        ax = fig.add_subplot(1,2,2)
        ax.plot(train_res_perplexity_smooth)
        ax.set_title('Smoothed Average codebook usage (perplexity).')
        ax.set_xlabel('iteration')

        fig.savefig(path)
        plt.close(fig)
