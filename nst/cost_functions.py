import torch
from nst.image_transformations import normalize_batch
from nst.configs.base_config import BaseNSTConfig

def compute_gram(matrix):
    '''
    Computes the gram matrix
    '''
    batches, channels, height, width = matrix.size()
    return (1/(channels * height * width)) * (torch.matmul(matrix.view(batches, channels, -1),
                                                torch.transpose(matrix.view(batches, channels, -1), 1, 2)))

def content_cost(config: BaseNSTConfig, input, target):
    # First normalize both the input and target (preprocess for VGG16)
    input_norm = normalize_batch(input)
    target_norm = normalize_batch(target)

    input_layers = config.forward_vgg(input_norm, config.content_layers.keys())
    target_layers = config.forward_vgg(target_norm, config.content_layers.keys())

    accumulated_loss = 0
    for layer, weight in enumerate(config.content_layers.values()):
        accumulated_loss = accumulated_loss + weight * torch.mean(torch.square(input_layers[layer] - target_layers[layer]))
    
    return accumulated_loss

def style_cost(config: BaseNSTConfig, input, target):
    # First normalize both the input and target (preprocess for VGG16)
    input_norm = normalize_batch(input)
    target_norm = normalize_batch(target)

    input_layers = config.forward_vgg(input_norm, config.style_layers.keys())
    target_layers = config.forward_vgg(target_norm, config.style_layers.keys())
    
    # The accumulated losses for the style
    accumulated_loss = 0
    for layer, weight in enumerate(config.style_layers.values()):
        accumulated_loss = accumulated_loss + weight * \
                            torch.mean(torch.square(compute_gram(input_layers[layer]) -
                                                    compute_gram(target_layers[layer])))
    
    return accumulated_loss

def total_variation_cost(input):
    tvloss = (
        torch.sum(torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:])) + 
        torch.sum(torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :]))
    )
    return tvloss

def total_cost(config: BaseNSTConfig, input, targets):
    # Weights
    REG_TV = 1e-6
    REG_STYLE = 1e6
    REG_CONTENT = 1.0
    
    # Extract content and style images
    content, style = targets
    
    # Get the content, style and tv variation losses
    closs = content_cost(config, input, content) * REG_CONTENT
    sloss = style_cost(config, input, style) * REG_STYLE
    tvloss = total_variation_cost(input) * REG_TV
        
    # Add it to the running list of losses
    config.content_losses.append(closs)
    config.style_losses.append(sloss)
    config.tv_losses.append(tvloss)
    
    print('****************************')
    print('Content Loss: {}'.format(closs.item()))
    print('Style Loss: {}'.format(sloss.item()))
    print('Total Variation Loss: {}'.format(tvloss.item()))
        
    # Apply the weights and add
    return closs + sloss + tvloss
