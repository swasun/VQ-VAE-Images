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
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', nargs='?', default=32, type=int, help='The size of the batch during training')
    parser.add_argument('--num_training_updates', nargs='?', default=25000, type=int, help='The number of updates during training')
    parser.add_argument('--num_hiddens', nargs='?', default=128, type=int, help='The number of hidden neurons in each layer')
    parser.add_argument('--num_residual_hiddens', nargs='?', default=32, type=int, help='The number of hidden neurons in each layer within a residual block')
    parser.add_argument('--num_residual_layers', nargs='?', default=2, type=int, help='The number of residual layers in a residual stack')
    parser.add_argument('--embedding_dim', nargs='?', default=64, type=int, help='Representing the dimensionality of the tensors in the quantized space')
    parser.add_argument('--num_embeddings', nargs='?', default=512, type=int, help='The number of vectors in the quantized space')
    parser.add_argument('--commitment_cost', nargs='?', default=0.25, type=float, help='Controls the weighting of the loss terms')
    parser.add_argument('--decay', nargs='?', default=0.99, type=float, help='Decay for the moving averages (set to 0 to not use EMA)')
    parser.add_argument('--learning_rate', nargs='?', default=3e-4, type=float, help='The learning rate of the optimizer during training updates')
    parser.add_argument('--use_kaiming_normal', nargs='?', default=True, type=bool, help='Use the weight normalization proposed in [He, K et al., 2015]')
    parser.add_argument('--data_path', nargs='?', default='data', type=str, help='The path of the data directory')
    parser.add_argument('--results_path', nargs='?', default='results', type=str, help='The path of the results directory')
    parser.add_argument('--loss_plot_name', nargs='?', default='loss.png', type=str, help='The file name of the training loss plot')
    parser.add_argument('--model_name', nargs='?', default='model.pth', type=str, help='The file name of trained model')
    parser.add_argument('--original_images_name', nargs='?', default='original_images.png', type=str, help='The file name of the original images used in evaluation')
    parser.add_argument('--validation_images_name', nargs='?', default='validation_images.png', type=str, help='The file name of the reconstructed images used in evaluation')
    parser.add_argument('--use_cuda_if_available', nargs='?', default=True, type=bool, help='Specify if GPU will be used if available')
    args = parser.parse_args()

    # Dataset and model hyperparameters
    configuration = Configuration.build_from_args(args)

    device = torch.device('cuda' if args.use_cuda_if_available and torch.cuda.is_available() else 'cpu') # Use GPU if cuda is available

    # Set the result path and create the directory if it doesn't exist
    results_path = '..' + os.sep + args.results_path
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    
    dataset_path = '..' + os.sep + args.data_path

    dataset = Cifar10Dataset(configuration.batch_size, dataset_path) # Create an instance of CIFAR10 dataset
    auto_encoder = AutoEncoder(device, configuration).to(device) # Create an AutoEncoder model using our GPU device

    optimizer = optim.Adam(auto_encoder.parameters(), lr=configuration.learning_rate, amsgrad=True) # Create an Adam optimizer instance
    trainer = Trainer(device, auto_encoder, optimizer, dataset) # Create a trainer instance
    trainer.train(configuration.num_training_updates) # Train our model on the CIFAR10 dataset
    trainer.save_loss_plot(results_path + os.sep + args.loss_plot_name) # Save the loss plot
    auto_encoder.save(results_path + os.sep + args.model_name) # Save our trained model

    evaluator = Evaluator(device, auto_encoder, dataset) # Create en Evaluator instance to evaluate our trained model
    evaluator.reconstruct() # Reconstruct our images from the embedded space
    evaluator.save_original_images_plot(results_path + os.sep + args.original_images_name) # Save the original images for comparaison purpose
    evaluator.save_validation_reconstructions_plot(results_path + os.sep + args.validation_images_name) # Reconstruct the decoded images and save them
