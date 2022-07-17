import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # For the transformation network, the authors only used 3x3 convolutions
        self.conv = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3
        )
        self.batch_norm = nn.InstanceNorm2d(self.out_channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        # First convolution
        orig_x = x.clone()
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        # Second convolution
        x = self.conv(x)
        x = self.batch_norm(x)

        # Now add the original to the new one (and use center cropping)
        # Calulate the different between the size of each feature (in terms
        # of height/width) to get the center of the original feature
        height_diff = orig_x.size()[2] - x.size()[2]
        width_diff = orig_x.size()[3] - x.size()[3]

        # Add the original to the new (complete the residual block)
        x = (
            x
            + orig_x[
                :,
                :,
                height_diff // 2 : (orig_x.size()[2] - height_diff // 2),
                width_diff // 2 : (orig_x.size()[3] - width_diff // 2),
            ]
        )

        return x
