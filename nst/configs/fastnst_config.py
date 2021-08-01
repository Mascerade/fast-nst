import torch
import numpy as np
import glob
from typing import Sequence, List
from PIL import Image
from nst.configs.base_config import BaseNSTConfig
from nst.models_architectures.transformation_network import ImageTransformationNetwork
from nst.models_architectures.forward_vgg import ForwardVGG19

class FastNSTConfig(BaseNSTConfig):
    def __init__(self,
                 img_dim: Sequence[int],
                 content_imgs_path: str,
                 style_img_path: str,
                 val_split: int,
                 test_split: int,
                 batch_size: int):
        
        super(FastNSTConfig, self).__init__(img_dim, style_img_path)

        # Initialize variables
        self.content_imgs_path = content_imgs_path
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size

        # Create data
        self.data: List[str] = list(glob.iglob(self.content_imgs_path))
        self.total_data_size: int = len(self.data)
        self.train_size: int = len(self.data) - (val_split + test_split)

        # The transformation network
        self.transformation_net = ImageTransformationNetwork()
