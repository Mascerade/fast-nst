import torch
import numpy as np
import glob
from typing import Sequence, List
from PIL import Image
from nst.configs.base_config import BaseNSTConfig
from nst.models_architectures.transformation_network import ImageTransformationNetwork
from nst.models_architectures.forward_vgg import ForwardVGG19

class NSTConfig(BaseNSTConfig):
    def __init__(self,
                 img_dim: Sequence[int],
                 content_img_path: str,
                 style_img_path: str):
        super(NSTConfig, self).__init__(img_dim, style_img_path)
        
        # Initialize variables
        self.content_img_path = content_img_path

        # For plotting the losses over time
        self.content_losses: List[float] = []
        self.style_losses: List[float] = []
        self.tv_losses: List[float] = []