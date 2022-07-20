from PIL import Image
import numpy as np


def normalize_batch(batch):
    """
    Before we send an image into the VGG19, we have to normalize it. Takes  a tensor
    """
    # Define the means and standard deviations
    vgg_means = [0.485, 0.456, 0.406]
    vgg_std = [0.229, 0.224, 0.225]

    # Clone the batch to make changes to it
    ret = batch.clone()

    # Normalize to between 0 and 255 (input image is 255-value images, not floats)
    ret = ret / 255.0

    # Subtract the means and divide by the standard deviations
    ret[:, 0, :, :] = (ret[:, 0, :, :] - vgg_means[0]) / vgg_std[0]
    ret[:, 1, :, :] = (ret[:, 1, :, :] - vgg_means[1]) / vgg_std[1]
    ret[:, 2, :, :] = (ret[:, 2, :, :] - vgg_means[2]) / vgg_std[2]
    return ret


def add_noise(batch):
    """
    For the input image, we have to add noise so that the loss between the content image and
    input image is not 0. Takes a numpy array
    """
    mean = 0.0
    std = 10.0
    ret = batch + np.random.normal(mean, std, batch.shape)
    ret = np.clip(batch, 0, 255)
    return ret


def upsample(img, dim):
    img = Image.fromarray(img[0].transpose(1, 2, 0).astype("uint8"))
    img = np.asarray(img.resize(dim, resample=0))
    img = img.transpose(2, 0, 1).reshape(1, 3, *dim)
    return img
