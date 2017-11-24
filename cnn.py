import numpy as np

from torch.autograd import Variable
from torch import optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import argparse
import os
import shutil
import time
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class basicCNN(nn.Module):
	def __init__(self):
			super(basicCNN,self).__init__()
			self.conv1 = nn.Conv2d(3, 5, 3)
			self.conv2 = nn.Conv2d(5, 10, 3)
			self.fc1 = nn.Linear(640, 100)
			self.fc2 = nn.Linear(100, 4)
			self.out1=None
			self.out2=None
	
	def forward(self, x):
		out=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
		self.out1=out
		out=F.max_pool2d(F.relu(self.conv2(out)),2)
		self.out2=out
		out=out.view(-1,self.num_flat_features(out))
		out=F.relu(self.fc1(out))
		out=self.fc2(out)
		#out=F.relu(self.fc2(out))
		#out=self.fc3(out)
		return out

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features


net = basicCNN()
#print(list(net.parameters()))
params = list(net.parameters())
print(len(params))
print(params[0].size())