import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.augmentation as K


# Color augmentation module using Kornia
class ColorAugmentor(nn.Module):
    """
    Apply a rich set of color and style augmentations using Kornia.
    """

    def __init__(self,
                 brightness=(0.6, 1.4),
                 contrast=(0.6, 1.4),
                 saturation=(0.6, 1.4),
                 hue=(-0.1, 0.1),
                 p=0.5):
        super().__init__()
        # Compose multiple Kornia augmentations
        self.augment = nn.Sequential(
            K.ColorJitter(brightness=brightness,
                          contrast=contrast,
                          saturation=saturation,
                          hue=hue,
                          p=p),
        )

    def forward(self, image):
        return self.augment(image)
