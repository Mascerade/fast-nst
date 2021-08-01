import torch
import numpy as np
from typing import Sequence, List
from PIL import Image
from nst.models_architectures.transformation_network import ImageTransformationNetwork
from nst.models_architectures.forward_vgg import ForwardVGG19


class BaseNSTConfig():
    def __init__(self,
                 img_dim: Sequence[int],
                 style_img_path: str):
        # Initialize variables
        self.img_dim = img_dim
        self.style_img_path = style_img_path

        # Create data
        self.style_img = np.asarray(Image.open(self.style_img_path).resize(self.img_dim)).transpose(2, 0, 1)[0:3]
        self.style_img = self.style_img_np.reshape(1, 3, self.img_dim[0], self.img_dim[1])
        self.style_img_tensor = torch.from_numpy(np.copy(self.style_img)).float()

        # For plotting the losses over time
        self.content_losses: List[float] = []
        self.style_losses: List[float] = []
        self.tv_losses: List[float] = []
        
        # The VGG network
        self.forward_vgg = ForwardVGG19()