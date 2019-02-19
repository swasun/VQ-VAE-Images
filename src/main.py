 #####################################################################################
 # MIT License                                                                       #
 #                                                                                   #
 # Copyright (C) 2019 Charly Lamothe                                                 #
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

from auto_encoder import AutoEncoder
from trainer import Trainer
from evaluator import Evaluator
from cifar10_dataset import Cifar10Dataset
from configuration import Configuration

import torch
import torch.optim as optim
import os


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if cuda is available
    use_ema = True

    # Set the result path and create the directory if it doesn't exist
    results_path = '..' + os.sep + 'results'
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    
    dataset_path = '..' + os.sep + 'data'

    configuration = Configuration(decay=0.99 if use_ema else 0.0) # Get the dataset and model hyperparameters
    dataset = Cifar10Dataset(configuration.batch_size, dataset_path) # Create an instance of CIFAR10 dataset
    auto_encoder = AutoEncoder(device, configuration).to(device) # Create an AutoEncoder model using our GPU device

    optimizer = optim.Adam(auto_encoder.parameters(), lr=configuration.learning_rate, amsgrad=True) # Create an Adam optimizer instance
    trainer = Trainer(device, auto_encoder, optimizer, dataset) # Create a trainer instance
    trainer.train(configuration.num_training_updates) # Train our model on the CIFAR10 dataset
    trainer.save_loss_plot(results_path + os.sep + 'loss_ema_he-et-al.png') # Save the loss plot
    auto_encoder.save(results_path + os.sep + 'model_ema_he-et-al.pth') # Save our trained model

    evaluator = Evaluator(device, auto_encoder, dataset) # Create en Evaluator instance to evaluate our trained model
    evaluator.reconstruct() # Reconstruct our images from the embedded space
    evaluator.save_original_images_plot(results_path + os.sep + 'original_images_ema_he-et-al.png') # Save the original images for comparaison purpose
    evaluator.save_validation_reconstructions_plot(results_path + os.sep + 'validation_images_ema_he-et-al.png') # Reconstruct the decoded images and save them
