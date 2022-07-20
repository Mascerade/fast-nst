from typing import Tuple
from math import sqrt
import numpy as np
import torch
from PIL import Image


class ImageContainer:
    """
    Makes managing the image tensor easier
    """

    def __init__(
        self,
        image_path: str,
        img_dim: Tuple[int, int],
        low_res: bool = False,
        low_res_img_dim=(500, 500),
        requires_grad=False,
    ):
        self.img_dim = img_dim
        pil_image = Image.open(image_path).resize(self.img_dim)

        if low_res:
            # Need to use a scale factor in order to maintain the aspect ratio of the original image
            target_low_res_img_pixels = low_res_img_dim[0] * low_res_img_dim[1]
            # Calculate a constant to scale the image down
            assert self.img_dim[0] * self.img_dim[1] > target_low_res_img_pixels
            scale_constant = sqrt(
                target_low_res_img_pixels / (self.img_dim[0] * self.img_dim[1])
            )

            # Generate the low resolution content image
            self.img_dim = (
                int(self.img_dim[0] * scale_constant),
                int(self.img_dim[1] * scale_constant),
            )

        # Create the image tensor from Numpy
        np_image = np.asarray(pil_image).transpose(2, 0, 1).reshape(1, 3, *self.img_dim)
        self.image_tensor = torch.from_numpy(np.copy(np_image)).float()
        self.image_tensor.requires_grad = requires_grad

    def to_image(self):
        """
        Convert the tensor to a PIL image
        """
        return Image.fromarray(
            self.image_tensor.detach()
            .numpy()[0]
            .transpose(1, 2, 0)
            .astype("uint8")  # Channels have to be last
        )

    def upsample(self, img_dim):
        """
        Upsample the tensor to its orignal dimensions
        """
        img = Image.fromarray(
            self.image_tensor.detach().numpy()[0].transpose(1, 2, 0).astype("uint8")
        )
        img = np.asarray(img.resize(img_dim, resample=0))
        img = img.transpose(2, 0, 1).reshape(1, 3, *img_dim)
        self.image_tensor = torch.from_numpy(img).float()
