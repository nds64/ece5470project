import numpy as np
from scipy import misc
import os
import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
from sklearn.datasets import load_digits
import cnn
import time
# from keras.models import Sequential
# from keras.layers import Activation,Conv2D,MaxPooling2D,Flatten

# xTr=[]
# current_class=0
# yTr=[]
# FILEPATH='ece5470project/re_FishImages/combine_image/'
# for files in os.listdir(FILEPATH):
# 	for image in os.listdir(FILEPATH+files):
# 		img = misc.imread(FILEPATH+files+'/'+image)
# 		# if xTr==[]:
# 		# 	xTr=img
# 		# else:
# 		# 	xTr=np.concatenate((xTr,img),axis=)
# 		xTr.append(img)
# 		yTr.append(current_class)
# 	current_class+=1

# with open ('Y.p','w') as f:
# 	np.save(f,yTr)

# with open('X.p','w') as f:
# 	np.save(f,xTr)

with open ('Y.p','r') as f:
	trY = np.load(f)
with open('X.p','r') as f:
	trX = np.load(f)

n_examples,w,h,c = np.asarray(trX).shape
p = np.random.permutation(n_examples)

p = np.random.permutation(len(trX))
trX=trX[p]
trY=trY[p]

test_length	= 2000
	
teX=trX[-test_length:]
teY=trY[-test_length:]

trX=trX[:-test_length]
trY=trY[:-test_length]

print(teX.shape)
print(trX.shape)

def counts(y):
	c=[0,0,0,0]
	for i in y:
		c[i]+=1
	return c

print(counts(teY))

trX = torch.from_numpy(trX).float()
teX = torch.from_numpy(teX).float()
trY = torch.from_numpy(trY).long()
teY = torch.from_numpy(teY).long()



num_classes=4

def train(model, loss, optimizer, x_val, y_val):
    x = Variable(x_val, requires_grad=False)
    y = Variable(y_val, requires_grad=False)

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    fx = model.forward(x)
    output = loss.forward(fx, y)

    # Backward
    output.backward()

    # Update parameters
    optimizer.step()

    return output.data[0]

#model = build_model(n_features, n_classes)
model=cnn.basicCNN()

loss = torch.nn.CrossEntropyLoss(size_average=True)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
batch_size = 1
s=time.time()

for i in range(100):
	print('hello')
	cost = 0.
	num_batches = n_examples // batch_size
	for k in range(num_batches):
		start, end = k * batch_size, (k + 1) * batch_size
		cost += train(model, loss, optimizer,
				trX[start:end], trY[start:end])
	print("Epoch %d, cost = %f"
		      % (i + 1, cost / num_batches))
e=time.time()