import torch
import numpy as np
import glob
from typing import List, Tuple, Dict
from PIL import Image
from nst.configs.base_config import BaseNSTConfig
from nst.models_architectures.transformation_network import ImageTransformationNetwork
from nst.image_transformations import add_noise, normalize_batch


class FastNSTConfig(BaseNSTConfig):
    def __init__(
        self,
        name: str,
        content_imgs_path: str,
        val_split: int,
        epochs: int,
        batch_size: int,
        img_dim: Tuple[int, int],
        style_img_path: str,
        content_layers: Dict[int, float],  # Key is the layer, value is the weight
        style_layers: Dict[int, float],  # Key is the lyaer, value is the weight
        lr: float,
        optimizer,
        content_weight=1.0,
        style_weight=1e6,
        tv_weight=1e-6,
    ):
        batches = 0
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

        # Initialize variables
        self.content_imgs_path = content_imgs_path
        self.val_split = val_split
        self.batch_size = batch_size

        # Create data
        self.data: List[str] = list(glob.iglob(self.content_imgs_path))
        self.total_data_size: int = len(self.data)
        self.train_size: int = len(self.data) - val_split

        # The transformation network
        self.transformation_net = ImageTransformationNetwork()

        # Create precomputed style layers
        style_vgg = normalize_batch(self.style_img_ten)
        style_vgg = self.forward_vgg(style_vgg, self.style_layers.keys())
        self.precomputed_style = []
        for layer in style_vgg:
            self.precomputed_style.append(self.compute_gram(layer))

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
        if set_type == "train":
            if initial_pos + self.batch_size > self.train_size:
                batch_size = self.train_size - initial_pos

        # Make sure the batch is within the [MAX_TRAIN, MAX_VAL)
        elif set_type == "val":
            initial_pos = self.train_size + initial_pos
            if initial_pos + self.batch_size > self.train_size + self.val_split:
                batch_size = (self.train_size + self.val_split) - initial_pos

        # Make sure the batch is within the [MAX_VAL, TOTAL_DATA)
        elif set_type == "test":
            initial_pos = (self.train_size + self.val_split) + initial_pos
            if initial_pos + self.batch_size > self.total_data_size:
                batch_size = self.total_data_size - initial_pos

        for f in self.data[initial_pos : initial_pos + batch_size]:
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

    def train(self):
        opt = self.optimizer([self.transformation_net.parameters()], self.lr)
        for epoch in self.epochs:
            for batch, _ in enumerate(
                range(0, len(self.data) - self.val_split, self.batch_size)
            ):
                # Train batch has noise while the content batch is the actual image
                train_batch = self.load_training_batch(batch, "train")
                content_batch = np.copy(train_batch)

                # Add noise to the training batch
                train_batch = add_noise(train_batch)

                # Convert the batches to tensors
                train_batch = torch.from_numpy(train_batch).float()
                content_batch = torch.from_numpy(content_batch).float()

                # Zero the gradients
                opt.zero_grad()

                # Forward propagate
                gen_images = self.transformation_net(train_batch)

                # Compute loss
                loss = self.total_cost(
                    gen_images,
                    [content_batch, self.style_img_ten, [None, self.precomputed_style]],
                )

                # Backprop
                loss.backward()

                # Clip the gradient to minimize chance of exploding gradients
                torch.nn.utils.clip_grad_norm_(
                    self.transformation_net.parameters(), 1.0
                )

                # Apply gradients
                opt.step()

                print(f"Training Batch: {batch + 1} Loss: {loss}")
                print("****************************")

                if batch % 100 == 99:
                    torch.save(
                        self.transformation_net.state_dict(),
                        f"models/{self.name}/epoch_{epoch + 1}batch_{batch + 1}.pt",
                    )

            # Change the network to eval to do the validation
            self.transformation_net.eval()

            # Iterate through the validation set
            for batch, _ in enumerate(
                range(len(self.data) - self.val_split, len(self.data), self.batch_size)
            ):
                # Train batch has noise while the content batch is the actual image
                val_batch = self.load_training_batch(batch, self.batch_size, "val")
                content_batch = np.copy(val_batch)

                # Add noise to the training batch
                val_batch = add_noise(val_batch)

                # Convert the batches to tensors
                val_batch = torch.from_numpy(val_batch).float()
                content_batch = torch.from_numpy(content_batch).float()

                # Forward propagate
                gen_images = self.transformation_net(val_batch)

                # Compute loss
                loss = self.total_cost(
                    gen_images,
                    [content_batch, self.style_img_ten, [None, self.precomputed_style]],
                )

                print(f"Validation Batch: {batch + 1} Loss: {loss}")
