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

    input_layers = config.forward_vgg(input_norm, False)
    target_layers = config.forward_vgg(target_norm, False)

    accumulated_loss = 0
    for layer in range(len(input_layers)):
        accumulated_loss = accumulated_loss + torch.mean(torch.square(input_layers[layer] - target_layers[layer]))
    
    return accumulated_loss

def style_cost(config: BaseNSTConfig, input, target):
    # First normalize both the input and target (preprocess for VGG16)
    input_norm = normalize_batch(input)
    target_norm = normalize_batch(target)

    input_layers = config.forward_vgg(input_norm, True)
    target_layers = config.forward_vgg(target_norm, True)
    
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

def total_cost(config: BaseNSTConfig, input, targets):
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
    config.content_losses.append(closs)
    config.style_losses.append(sloss)
    config.tv_losses.append(tvloss)
    
    print('****************************')
    print('Content Loss: {}'.format(closs.item()))
    print('Style Loss: {}'.format(sloss.item()))
    print('Total Variation Loss: {}'.format(tvloss.item()))
        
    # Apply the weights and add
    return closs + sloss + tvloss
