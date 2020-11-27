import torch
import torch.optim as optim
import numpy as np
from src.common import Common
from src.training_functions import total_cost, load_training_batch
from src.image_transformations import add_noise

# Create the optimizer
opt = optim.Adam(Common.transformation_net.parameters(), lr=1e-3)

for epoch in range(2):
    Common.transformation_net.train()
    for batch, _ in enumerate(range(0, Common.MAX_TRAIN, Common.BATCH_SIZE)):
        # Skip what we've already done
        if epoch == 0 and batch < 0:
            continue
        
        # The content batch is the same as the train batch, except train batch has noise added to it
        train_batch = load_training_batch(batch, Common.BATCH_SIZE, 'train')
        content_batch = np.copy(train_batch)

        # Add noise to the training batch
        train_batch = add_noise(train_batch)

        # Convert the batches to tensors
        train_batch = torch.from_numpy(train_batch).float()
        content_batch = torch.from_numpy(content_batch).float()

        # Zero the gradients
        opt.zero_grad()

        # Forward propagate
        gen_images = Common.transformation_net(train_batch)

        # Compute loss
        loss = total_cost(gen_images, [content_batch, Common.STYLE_IMAGE_TENSOR])

        # Backprop
        loss.backward()

        # Clip the gradient to minimize chance of exploding gradients
        torch.nn.utils.clip_grad_norm_(Common.transformation_net.parameters(), 1.0)

        # Apply gradients
        opt.step()

        print("Training Batch: {}".format(batch + 1), "Loss: {:f}".format(loss))
        print('****************************')
    
    # Change the network to eval to do the validation
    Common.transformation_net.eval()

    # Iterate through the validation set
    for batch, _ in enumerate(range(Common.MAX_TRAIN, Common.MAX_VAL, Common.BATCH_SIZE)):
        # The content batch is the same as the train batch, except train batch has noise added to it
        val_batch = load_training_batch(batch, Common.BATCH_SIZE, 'val')
        content_batch = np.copy(val_batch)
        
        # Add noise to the training batch
        val_batch = add_noise(val_batch)
        
        # Convert the batches to tensors
        val_batch = torch.from_numpy(val_batch).float()
        content_batch = torch.from_numpy(content_batch).float()
        
        # Forward propagate
        gen_images = Common.transformation_net(val_batch)

        # Compute loss
        loss = total_cost(gen_images, [content_batch, Common.STYLE_IMAGE_TENSOR])
        
        print("Validation Batch: {}".format(batch + 1), "Loss: {:f}".format(loss))