import torch
from torchvision.models import vgg19  # type: ignore


class ForwardVGG19(torch.nn.Module):
    def __init__(self):
        super(ForwardVGG19, self).__init__()
        vgg = vgg19(pretrained=True)
        vgg.eval()
        self.features = vgg.features
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x, layers):
        results = []
        for i, model in enumerate(self.features):
            x = model(x)

            if i in layers:
                results.append(x)

        return results
