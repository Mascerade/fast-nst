import torch
import numpy as np
import glob
from typing import Sequence, List, Dict
from PIL import Image
from nst.configs.base_config import BaseNSTConfig
from nst.models_architectures.transformation_network import ImageTransformationNetwork
from nst.models_architectures.forward_vgg import ForwardVGG19
from nst.cost_functions import total_cost
from nst.plotting_images import plot_img

class NSTConfig(BaseNSTConfig):
    def __init__(self,
                 content_img_path: str,
                 *args,
                 **kwargs):
        super(NSTConfig, self).__init__(*args, **kwargs)

        # Initialize variables
        self.content_img_path = content_img_path
        
        # Note: The images are using 255 int
        self.content_img = np.asarray(Image.open(self.content_img_path).resize(self.img_dim)).transpose(2, 0, 1)[0:3]
        self.content_img = self.content_img.reshape(1, 3, *self.img_dim)
        self.content_img_ten = torch.from_numpy(self.content_img).float()
        self.target_img = torch.clone(self.content_img_ten)
        self.target_img.requires_grad = True

        self.opt = self.optimizer([self.target_img], self.lr)

        # For plotting the losses over time
        self.content_losses: List[float] = []
        self.style_losses: List[float] = []
        self.tv_losses: List[float] = []

    def train(self):
        for epoch in range(self.epochs):
            for batch in range(self.batches):
                # Zero the gradients
                self.opt.zero_grad()

                # Compute loss
                loss = total_cost(self,
                                  self.target_img,
                                  [self.content_img_ten, self.style_img_ten])

                # Backprop
                loss.backward()

                # Apply gradients
                self.opt.step()

                # Make sure the values are not more than 255 or less than 0
                self.target_img.data.clamp_(0, 255)

                # Every 20 batches, show the loss graphs and the image so far
                if (batch % 20 == 19):
                    #plot_losses()
                    plot_img(self.target_img.numpy())

                print("Epoch: {} Training Batch: {}".format(epoch + 1, batch + 1), "Loss: {:f}".format(loss))
                print('****************************')