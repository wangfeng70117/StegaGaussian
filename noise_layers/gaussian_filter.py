import torch.nn as nn
from kornia.filters import GaussianBlur2d


class GF(nn.Module):

	def __init__(self, sigma, kernel=7):
		super(GF, self).__init__()
		self.gaussian_filter = GaussianBlur2d((kernel, kernel), (sigma, sigma))

	def forward(self, image):
		return self.gaussian_filter(image)

