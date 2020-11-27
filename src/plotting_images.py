import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from src.common import Common
from src.training_functions import load_training_batch

def show_sample_images():
    _, axarr = plt.subplots(2,2)
    
    # Show colored images
    axarr[0,0].imshow(np.asarray(Image.open(Common.DATA[0]).resize(Common.IMG_DIMENSIONS)))
    axarr[0,1].imshow(np.asarray(Image.open(Common.DATA[4]).resize(Common.IMG_DIMENSIONS)))
    axarr[1,0].imshow(np.asarray(Image.open(Common.DATA[8]).resize(Common.IMG_DIMENSIONS)))
    
    # Grayscale example
    grayscale = np.asarray(Image.open(Common.DATA[13]).resize(Common.IMG_DIMENSIONS))
    grayscale = np.stack((grayscale, grayscale, grayscale)).transpose(1, 2, 0)
    axarr[1,1].imshow(grayscale)
    
def training_show_img():
    # Get an image from the validation set
    img = load_training_batch(0, 10, 'val')[4]
    
    # Convert to tensor
    train_img = torch.from_numpy(img.reshape(1, 3, 256, 256)).float()
    
    # Put through network
    gen_img = Common.transformation_net(train_img)
    gen_img = gen_img.detach().numpy()
    
    # Clip the floats
    gen_img = np.clip(gen_img, 0, 255)
    
    # Convert to ints (for images)
    gen_img = gen_img.astype('uint8')
    gen_img = gen_img.reshape(3, 256, 256).transpose(1, 2, 0)
    
    # Show the image
    plt.imshow(gen_img)
    plt.show()

def show_img(img):
    # Convert to tensor
    img = torch.from_numpy(img.reshape(1, 3, 256, 256)).float()
    
    # Put through network
    gen_img = Common.transformation_net(img)
    gen_img = gen_img.detach().numpy()
    
    # Clip the floats
    gen_img = np.clip(gen_img, 0, 255)
    
    # Convert to ints (for images)
    gen_img = gen_img.astype('uint8')
    gen_img = gen_img.reshape(3, 256, 256).transpose(1, 2, 0)
    
    # Show the image
    plt.imshow(gen_img)
    plt.show()

def plot_losses():
    # Print info about content losses
    plt.plot(Common.content_losses[int(len(Common.content_losses) * 0.0):])
    plt.show()
    print(Common.content_losses[len(Common.content_losses) - 1])

    # Print info about style losses
    plt.plot(Common.style_losses[int(len(Common.style_losses) * 0.0):])
    plt.show()
    print(Common.style_losses[len(Common.style_losses) - 1])

    # Print info about total variation losses
    plt.plot(Common.tv_losses[int(len(Common.tv_losses) * 0.0):])
    plt.show()
    print(Common.tv_losses[len(Common.tv_losses) - 1])
