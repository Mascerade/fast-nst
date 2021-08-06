import torch
import numpy as np
from typing import Sequence, Dict, List
from PIL import Image
import torch
from nst.image_transformations import normalize_batch
from nst.models_architectures.forward_vgg import ForwardVGG19


class BaseNSTConfig():
    def __init__(self,
                 img_dim: Sequence[int],
                 style_img_path: str,
                 content_layers: Dict[int, int],
                 style_layers: Dict[int, int],
                 epochs: int,
                 batches: int,
                 lr: int,
                 optimizer,
                 content_weight = 1.0,
                 style_weight = 1e6,
                 tv_weight = 1e-6):
        # Initialize variables
        self.img_dim = img_dim
        self.style_img_path = style_img_path
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.epochs = epochs
        self.batches = batches
        self.lr = lr
        self.optimizer = optimizer

        # Create data
        self.style_img = np.asarray(Image.open(self.style_img_path).resize(self.img_dim)).transpose(2, 0, 1)[0:3]
        self.style_img = self.style_img.reshape(1, 3, *self.img_dim)
        self.style_img_ten = torch.from_numpy(np.copy(self.style_img)).float()

        # For plotting the losses over time
        self.content_losses: List[float] = []
        self.style_losses: List[float] = []
        self.tv_losses: List[float] = []
        
        # The VGG network
        self.forward_vgg = ForwardVGG19()

    def compute_gram(self, matrix):
        '''
        Computes the gram matrix
        '''
        batches, channels, height, width = matrix.size()
        return (1/(channels * height * width)) * (torch.matmul(matrix.view(batches, channels, -1),
                                                    torch.transpose(matrix.view(batches, channels, -1), 1, 2)))

    def content_cost(self, input, target, precomputed):
        # First normalize both the input and target (preprocess for VGG16)
        input_norm = normalize_batch(input)
        input_layers = self.forward_vgg(input_norm, self.content_layers.keys())
        
        target_layers = precomputed
        if target_layers is None:
            target_norm = normalize_batch(target)
            target_layers = self.forward_vgg(target_norm, self.content_layers.keys())

        accumulated_loss = 0
        for layer, weight in enumerate(self.content_layers.values()):
            accumulated_loss = accumulated_loss + weight * torch.mean(torch.square(input_layers[layer] - target_layers[layer]))
        
        return accumulated_loss

    def style_cost(self, input, target, precomputed):
        # First normalize both the input and target (preprocess for VGG16)
        input_norm = normalize_batch(input)
        input_layers = self.forward_vgg(input_norm, self.style_layers.keys())
        
        target_layers = precomputed
        if target_layers is None:
            target_norm = normalize_batch(target)
            target_vgg_layers = self.forward_vgg(target_norm, self.style_layers.keys())
            target_layers = []
            for x in target_vgg_layers:
                target_layers.append(self.compute_gram(x))

        # The accumulated losses for the style
        accumulated_loss = 0
        for layer, weight in enumerate(self.style_layers.values()):
            accumulated_loss = accumulated_loss + weight * \
                                torch.mean(torch.square(self.compute_gram(input_layers[layer]) -
                                                        target_layers[layer]))
        
        return accumulated_loss

    def total_variation_cost(self, input):
        norm = input / 255.0
        tvloss = (
            torch.sum(torch.abs(norm[:, :, :, :-1] - norm[:, :, :, 1:])) + 
            torch.sum(torch.abs(norm[:, :, :-1, :] - norm[:, :, 1:, :]))
        )
        return tvloss

    def total_cost(self, input, targets):
        # Extract content and style images
        content, style, precomputed = targets
        
        # Get the content, style and tv variation losses
        closs = self.content_cost(input, content, precomputed[0]) * self.content_weight
        sloss = self.style_cost(input, style, precomputed[1]) * self.style_weight
        tvloss = self.total_variation_cost(input) * self.tv_weight
            
        # Add it to the running list of losses
        self.content_losses.append(closs)
        self.style_losses.append(sloss)
        self.tv_losses.append(tvloss)
        
        print('****************************')
        print('Content Loss: {}'.format(closs.item()))
        print('Style Loss: {}'.format(sloss.item()))
        print('Total Variation Loss: {}'.format(tvloss.item()))
            
        # Apply the weights and add
        return closs + sloss + tvloss
