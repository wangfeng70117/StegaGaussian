import torch
import torch.nn as nn
import kornia.augmentation as K

# class Rotation(nn.Module):
#     def __init__(self, degree):
#         super(Rotation, self).__init__()
#         self.degree = degree
#
#     def forward(self, image_and_cover):
#         image, cover_image = image_and_cover
#         # 生成 -1 或 1
#         rand_sign = (torch.randint(0, 2, (1,)) * 2 - 1).item()
#         angle = self.degree * rand_sign
#         # 每次 forward 动态创建 RandomAffine
#         rotation = K.RandomAffine(degrees=(angle, angle), p=1.0)
#         image = rotation(image)
#         return image

# class Rotation(nn.Module):
#     def __init__(self, degree):
#         super(Rotation, self).__init__()
#         self.degree = degree
#         # 初始化时不设置角度
#         self.rotation = K.RandomAffine(degrees=(0, 0), p=1.0)
#
#     def forward(self, image_and_cover):
#         image, cover_image = image_and_cover
#         # 生成 -1 或 1
#         rand_sign = (torch.randint(0, 2, (1,)) * 2 - 1).item()
#         angle = self.degree * rand_sign
#         # 修改 degrees 参数（注意：Kornia v0.6.9+ 支持设置）
#         self.rotation.degrees = (angle, angle)
#         image = self.rotation(image)
#         return image


import torch.nn as nn
import kornia.augmentation as K

class Rotation(nn.Module):
    def __init__(self, degree):
        super(Rotation, self).__init__()
        self.degree = degree
        self.rotation = K.RandomAffine(degrees=self.degree, p=1.0)



    def forward(self, image):
        return self.rotation(image)
