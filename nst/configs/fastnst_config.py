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

    def load_training_batch(self, current_batch: int, set_type: str):
        """
        Load different batches of data (essentially a custom data loader for training, validation, and testing)
        """
        # The initial position is where we want to start getting the batch
        # So it is the starting index of the batch
        initial_pos = current_batch * self.batch_size
        
        # List to store the images
        images = []
        
        # Make sure the batch is within the [0, self.train_size)
        if set_type == 'train':
            if initial_pos + self.batch_size > self.train_size:
                batch_size = self.train_size - initial_pos
        
        # Make sure the batch is within the [MAX_TRAIN, MAX_VAL)
        elif set_type == 'val':
            initial_pos = self.train_size + initial_pos
            if initial_pos + self.batch_size > self.train_size + self.val_split:
                batch_size = (self.train_size + self.val_split) - initial_pos
        
        # Make sure the batch is within the [MAX_VAL, TOTAL_DATA)
        elif set_type == 'test':
            initial_pos = (self.train_size + self.val_split) + initial_pos
            if initial_pos + self.batch_size > self.total_data_size:
                batch_size = self.total_data_size - initial_pos

        for f in self.data[initial_pos:initial_pos + batch_size]:
            # Resize the image to 256 x 256
            image = np.asarray(Image.open(f).resize(self.img_dim))
            
            # If the image is grayscale, stack the image 3 times to get 3 channels
            if image.shape == self.img_dim:
                image = np.stack((image, image, image))
                images.append(image)
                continue
                
            # Transpose the image to have channels first
            image = image.transpose(2, 0, 1)
            images.append(image)
        
        return np.array(images)
