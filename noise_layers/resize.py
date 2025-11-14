# import torch.nn as nn
# import torch.nn.functional as F
# import torch

# class Resize(nn.Module):
#     """
#     Resize the image. The target size is original size * resize_ratio
#     """
#     def __init__(self, resize_ratio_range, interpolation_method='nearest'):
#         super(Resize, self).__init__()
#         self.resize_ratio_min = resize_ratio_range[0]
#         self.resize_ratio_max = resize_ratio_range[1]
#         self.interpolation_method = interpolation_method
#
#
#     def forward(self, noised_and_cover):
#
#         resize_ratio = torch.empty(1).uniform_(self.resize_ratio_min, self.resize_ratio_max).item()
#         image, cover_image = noised_and_cover
#         image = F.interpolate(
#                                     image,
#                                     scale_factor=(resize_ratio, resize_ratio),
#                                     mode=self.interpolation_method)
#         print(f'resized image.shape is {image.shape}')
#         return image


import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class Resize(nn.Module):
    def __init__(self, resize_range=(0.6, 1.3), interpolation_method='bilinear'):
        super().__init__()
        self.resize_range = resize_range
        self.interpolation_method = interpolation_method

    def forward(self, image):
        B, C, H, W = image.shape

        scale = torch.empty(1, device=image.device).uniform_(*self.resize_range).item()

        new_H, new_W = int(H * scale), int(W * scale)

        resized = F.interpolate(image, size=(new_H, new_W), mode=self.interpolation_method)

        restored = F.interpolate(resized, size=(H, W), mode=self.interpolation_method)

        return restored
