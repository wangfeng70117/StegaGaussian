import torch.nn as nn
import kornia.augmentation as K


class GN(nn.Module):

	def __init__(self):
		super(GN, self).__init__()
		self.gaussian_noise = K.RandomGaussianNoise(mean=0.0, std=0.1, p=1.0)


	def forward(self, image):
		return self.gaussian_noise(image)
