import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt  # type: ignore
from nst.configs.base_config import BaseNSTConfig
from nst.configs.fastnst_config import FastNSTConfig


def show_sample_images(config: FastNSTConfig):
    _, axarr = plt.subplots(2, 2)

    # Show colored images
    axarr[0, 0].imshow(np.asarray(Image.open(config.data[0]).resize(config.img_dim)))
    axarr[0, 1].imshow(np.asarray(Image.open(config.data[4]).resize(config.img_dim)))
    axarr[1, 0].imshow(np.asarray(Image.open(config.data[8]).resize(config.img_dim)))

    # Grayscale example
    grayscale = np.asarray(Image.open(config.data[13]).resize(config.img_dim))
    grayscale = np.stack((grayscale, grayscale, grayscale)).transpose(1, 2, 0)
    axarr[1, 1].imshow(grayscale)


def training_show_img(config: FastNSTConfig):
    # Get an image from the validation set
    img = config.load_training_batch(0, "val")[4]

    # Convert to tensor
    train_img = torch.from_numpy(img.reshape(1, 3, *config.img_dim)).float()

    # Put through network
    gen_img = config.transformation_net(train_img)
    gen_img = gen_img.detach().numpy()

    # Clip the floats
    gen_img = np.clip(gen_img, 0, 255)

    # Convert to ints (for images)
    gen_img = gen_img.astype("uint8")
    gen_img = gen_img.reshape(3, 256, 256).transpose(1, 2, 0)

    # Show the image
    plt.imshow(gen_img)
    plt.show()


def plot_img(img):
    img = img[0].transpose(1, 2, 0).astype("uint8")
    plt.figure(figsize=(10, 5))
    plt.imshow(img)
    plt.show()


def show_img(config: FastNSTConfig, img):
    # Convert to tensor
    img = torch.from_numpy(img.reshape(1, 3, *config.img_dim)).float()

    # Put through network
    gen_img = config.transformation_net(img)
    gen_img = gen_img.detach().numpy()

    # Clip the floats
    gen_img = np.clip(gen_img, 0, 255)

    # Convert to ints (for images)
    gen_img = gen_img.astype("uint8")
    gen_img = gen_img.reshape(3, *config.img_dim).transpose(1, 2, 0)

    # Show the image
    plt.imshow(gen_img)
    plt.show()


def plot_losses(config: BaseNSTConfig):
    """
    Plots the losses on a graph.
    *NOTE: Only use if in a Jupyter Notebook
    """

    # Print info about content losses
    plt.plot(config.content_losses[int(len(config.content_losses) * 0.0) :])
    plt.show()
    print(config.content_losses[len(config.content_losses) - 1])

    # Print info about style losses
    plt.plot(config.style_losses[int(len(config.style_losses) * 0.0) :])
    plt.show()
    print(config.style_losses[len(config.style_losses) - 1])

    # Print info about total variation losses
    plt.plot(config.tv_losses[int(len(config.tv_losses) * 0.0) :])
    plt.show()
    print(config.tv_losses[len(config.tv_losses) - 1])
