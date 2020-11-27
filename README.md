## Introduction
Fast Neural Style Transfer is a method proposed by Justin Johnson, Alexandre Alahi, and Li Fei-Fei in their paper [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf). It is an adaptation of Gatys et. al.'s [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576.pdf) where an image is transformed to be in the style of another. Although producing great images, it takes a long time to train and it optimizes on only one image. With Johnson's method, we are able to create a neural network that can convert any image into another style that it was trained on in seconds.

## Structure
There is a notebook that has the entire network laid out nicely so that someone could simply look at it and understand how it works. For the Python files, this is the layout:  
`src` -  Contains all the different functions necessary functions needed  
> --> `models_architectures` - Has the different models used (residual block, VGG19 and the actual transformation network)

## Training
In order to train, all you have to do is change the style image within `common.py` in `src` and then run train! If you want, you can also adjust the content and style regularizers within `training_functions.py` in the `total_cost` function.

## To Do
* Save generated images
* Add sample images here
* Add commands to make training/inferencing more user friendly