import torch
import math
import numpy as np
from typing import List, Tuple, Dict
from PIL import Image
from nst.configs.base_config import BaseNSTConfig
from nst.plotting_images import plot_img, save_img
from nst.image_transformations import normalize_batch, upsample


class GatysNSTConfig(BaseNSTConfig):
    def __init__(
        self,
        name: str,
        content_img_path: str,
        high_res: bool,
        img_dim: Tuple[int, int],
        style_img_path: str,
        content_layers: Dict[int, float],
        style_layers: Dict[int, float],
        epochs: int,
        batches: int,
        lr: float,
        optimizer,
        content_weight=1.0,
        style_weight=1e6,
        tv_weight=1e-6,
    ):
        super().__init__(
            name,
            img_dim,
            style_img_path,
            content_layers,
            style_layers,
            epochs,
            batches,
            lr,
            optimizer,
            content_weight,
            style_weight,
            tv_weight,
        )
        self.target_low_res_img_pixels = 500 * 500

        # Initialize variables
        self.content_img_path = content_img_path
        self.high_res = high_res

        # Note: The images are using 255 int
        # The transpose is used because PIL Image uses channels last while vgg uses channels first
        self.content_img = np.asarray(
            Image.open(self.content_img_path).resize(self.img_dim)
        ).transpose(2, 0, 1)
        self.content_img = self.content_img.reshape(1, 3, *self.img_dim)
        self.content_img_ten = torch.from_numpy(np.copy(self.content_img)).float()
        if not self.high_res:
            self.target_img = torch.clone(self.content_img_ten)
            self.target_img.requires_grad = True
        else:
            # Calculate a constant to scale the image down
            assert self.img_dim[0] * self.img_dim[1] > self.target_low_res_img_pixels
            scale_constant = math.sqrt(
                self.target_low_res_img_pixels / (self.img_dim[0] * self.img_dim[1])
            )

            # Generate the low resolution content image
            self.low_res_img_dim = (
                int(self.img_dim[0] * scale_constant),
                int(self.img_dim[1] * scale_constant),
            )
            self.low_res_content_img = np.asarray(
                Image.open(self.content_img_path).resize(self.low_res_img_dim)
            ).transpose(2, 0, 1)
            self.low_res_content_img = self.low_res_content_img.reshape(
                1, 3, *self.low_res_img_dim
            )
            self.low_res_content_img_ten = torch.from_numpy(
                np.copy(self.low_res_content_img)
            ).float()

            # Copy the low resolution content image for the initial target/input image
            self.low_res_target_img = torch.clone(self.low_res_content_img_ten)
            self.low_res_target_img.requires_grad = True

            # Create the low resolution style image
            self.low_res_style_img = np.asarray(
                Image.open(self.style_img_path).resize(self.low_res_img_dim)
            ).transpose(2, 0, 1)
            self.low_res_style_img = self.low_res_style_img.reshape(
                1, 3, *self.low_res_img_dim
            )
            self.low_res_style_img_ten = torch.from_numpy(
                np.copy(self.low_res_style_img)
            ).float()

            # Create precomputed low res style layers
            style_vgg = normalize_batch(self.low_res_style_img_ten)
            style_vgg = self.forward_vgg(style_vgg, self.style_layers.keys())
            self.low_res_precomputed_style = []
            for layer in style_vgg:
                self.low_res_precomputed_style.append(self.compute_gram(layer))

            # Create precomputed low res content layers
            self.low_res_precomputed_content = normalize_batch(
                self.low_res_content_img_ten
            )
            self.low_res_precomputed_content = self.forward_vgg(
                self.low_res_precomputed_content, self.content_layers.keys()
            )

            self.low_res_precomputed = [
                self.low_res_precomputed_content,
                self.low_res_precomputed_style,
            ]

        # For plotting the losses over time
        self.content_losses: List[float] = []
        self.style_losses: List[float] = []
        self.tv_losses: List[float] = []

        # Create precomputed style layers
        style_vgg = normalize_batch(self.style_img_ten)
        style_vgg = self.forward_vgg(style_vgg, self.style_layers.keys())
        self.precomputed_style = []
        for layer in style_vgg:
            self.precomputed_style.append(self.compute_gram(layer))

        # Create precomputed content layers
        self.precomputed_content = normalize_batch(self.content_img_ten)
        self.precomputed_content = self.forward_vgg(
            self.precomputed_content, self.content_layers.keys()
        )

        self.precomputed = [self.precomputed_content, self.precomputed_style]

    def train(self):
        if self.high_res:
            self.train_low_res()
        opt = self.optimizer([self.target_img], self.lr)
        self.train_img(
            self.name,
            opt,
            self.target_img,
            self.content_img_ten,
            self.style_img_ten,
            self.precomputed,
        )

    def train_low_res(self):
        opt = self.optimizer([self.low_res_target_img], self.lr)
        self.train_img(
            self.name + "_low_res",
            opt,
            self.low_res_target_img,
            self.low_res_content_img,
            self.low_res_style_img,
            self.low_res_precomputed,
        )
        self.target_img = upsample(
            self.low_res_target_img.detach().numpy(), self.img_dim
        )
        self.target_img = torch.from_numpy(self.target_img)
        self.target_img.requires_grad = True

    def train_img(
        self, folder: str, opt, init_img, content_img, style_img, precomputed
    ):
        for epoch in range(self.epochs):
            for batch in range(self.batches):
                # Zero the gradients
                opt.zero_grad()

                # Compute loss
                loss = self.total_cost(init_img, [content_img, style_img, precomputed])

                # Backprop
                loss.backward()

                # Apply gradients
                opt.step()

                # Make sure the values are not more than 255 or less than 0
                init_img.data.clamp_(0, 255)

                # Every 20 batches, show the loss graphs and the image so far
                if batch % 20 == 19:
                    # plot_losses()
                    # plot_img(init_img.detach().numpy())
                    save_img(
                        init_img.detach().numpy(),
                        folder,
                        f"epoch_{epoch}_batch_{batch}",
                    )

                print(
                    "Epoch: {} Training Batch: {}".format(epoch + 1, batch + 1),
                    "Loss: {:f}".format(loss),
                )
                print("****************************")
