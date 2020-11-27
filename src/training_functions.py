import torch
import numpy as np
from PIL import Image
from src.image_transformations import normalize_batch
from src.common import Common

def compute_gram(matrix):
    '''
    Computes the gram matrix
    '''
    batches, channels, height, width = matrix.size()
    return (1/(channels * height * width)) * (torch.matmul(matrix.view(batches, channels, -1),
                                                torch.transpose(matrix.view(batches, channels, -1), 1, 2)))

def content_cost(input, target):
    # First normalize both the input and target (preprocess for VGG16)
    input_norm = normalize_batch(input)
    target_norm = normalize_batch(target)

    input_layers = Common.forward_vgg(input_norm, False)
    target_layers = Common.forward_vgg(target_norm, False)

    accumulated_loss = 0
    for layer in range(len(input_layers)):
        accumulated_loss = accumulated_loss + torch.mean(torch.square(input_layers[layer] - target_layers[layer]))
    
    return accumulated_loss

def style_cost(input, target):
    # First normalize both the input and target (preprocess for VGG16)
    input_norm = normalize_batch(input)
    target_norm = normalize_batch(target)

    input_layers = Common.forward_vgg(input_norm, True)
    target_layers = Common.forward_vgg(target_norm, True)
    
    # layer weights
    layer_weights = [0.3, 0.7, 0.7, 0.3]
    
    # The accumulated losses for the style
    accumulated_loss = 0
    for layer in range(len(input_layers)):
        accumulated_loss = accumulated_loss + layer_weights[layer] * \
                            torch.mean(torch.square(compute_gram(input_layers[layer]) -
                                                    compute_gram(target_layers[layer])))
    
    return accumulated_loss

def total_variation_cost(input):
    tvloss = (
        torch.sum(torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:])) + 
        torch.sum(torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :]))
    )
    return tvloss

def total_cost(input, targets):
    # Weights
    REG_TV = 1e-6
    REG_STYLE = 1e6
    REG_CONTENT = 1.0
    
    # Extract content and style images
    content, style = targets
    
    # Get the content, style and tv variation losses
    closs = content_cost(input, content) * REG_CONTENT
    sloss = style_cost(input, style) * REG_STYLE
    tvloss = total_variation_cost(input) * REG_TV
        
    # Add it to the running list of losses
    Common.content_losses.append(closs)
    Common.style_losses.append(sloss)
    Common.tv_losses.append(tvloss)
    
    print('****************************')
    print('Content Loss: {}'.format(closs.item()))
    print('Style Loss: {}'.format(sloss.item()))
    print('Total Variation Loss: {}'.format(tvloss.item()))
        
    # Apply the weights and add
    return closs + sloss + tvloss

def load_training_batch(current_batch, batch_size, set_type):
    """
    Load different batches of data (essentially a custom data loader for training, validation, and testing)
    """
    # The initial position is where we want to start getting the batch
    # So it is the starting index of the batch
    initial_pos = current_batch * batch_size
    
    # List to store the images
    images = []
    
    # Make sure the batch is within the [0, MAX_TRAIN]
    if set_type == 'train':
        if initial_pos + batch_size > Common.MAX_TRAIN:
            batch_size = Common.MAX_TRAIN - initial_pos
    
    # Make sure the batch is within the [MAX_TRAIN, MAX_VAL]
    elif set_type == 'val':
        initial_pos = Common.MAX_TRAIN + initial_pos
        if initial_pos + batch_size > Common.MAX_VAL:
            batch_size = Common.MAX_VAL - initial_pos
    
    # Make sure the batch is within the [MAX_VAL, TOTAL_DATA]
    elif set_type == 'test':
        initial_pos = Common.MAX_VAL + initial_pos
        if initial_pos + batch_size > Common.TOTAL_DATA:
            batch_size = Common.TOTAL_DATA - initial_pos

    for f in Common.DATA[initial_pos:initial_pos + batch_size]:
        # Resize the image to 256 x 256
        image = np.asarray(Image.open(f).resize(Common.IMG_DIMENSIONS))
        
        # If the image is grayscale, stack the image 3 times to get 3 channels
        if image.shape == Common.IMG_DIMENSIONS:
            image = np.stack((image, image, image))
            images.append(image)
            continue
            
        # Transpose the image to have channels first
        image = image.transpose(2, 0, 1)
        images.append(image)
    
    return np.array(images)
