import torch
import numpy as np
import glob
from PIL import Image
from src.models_architectures.transformation_network import ImageTransformationNetwork
from src.models_architectures.forward_vgg import ForwardVGG19

class Common():
    # Constants
    IMG_DIMENSIONS = (256, 256)
    DATA = list(glob.iglob('data/content_images/*'))
    STYLE_IMAGE = np.asarray(Image.open('data/starry_night.jpeg').resize(IMG_DIMENSIONS)).transpose(2, 0, 1)[0:3]

    # Make the style image a batch and convert
    STYLE_IMAGE = STYLE_IMAGE.reshape(1, 3, 256, 256)
    STYLE_IMAGE_TENSOR = torch.from_numpy(np.copy(STYLE_IMAGE)).float()
    MAX_TRAIN = 80000
    MAX_VAL = 81000
    TOTAL_DATA = len(DATA)
    BATCH_SIZE = 4

    # For plotting the losses over time
    content_losses = []
    style_losses = []
    tv_losses = []

    # The transformation network
    transformation_net = ImageTransformationNetwork()
    
    # The VGG network
    forward_vgg = ForwardVGG19()