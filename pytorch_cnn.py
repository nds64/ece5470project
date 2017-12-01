import numpy as np
from scipy import misc
import os
import torch
from torch.autograd import Variable
from torch import optim
from sklearn.datasets import load_digits
import cnn
import time

xTr=[]
current_class=0
yTr=[]
FILEPATH='re_FishImages/re_fish_image/'
for files in os.listdir(FILEPATH):
	for image in os.listdir(FILEPATH+files):
		img = misc.imread(FILEPATH+files+'/'+image)
		#print img.shape
		# if xTr==[]:
		# 	xTr=img
		# else:
		# 	xTr=np.concatenate((xTr,img),axis=)
		xTr.append(img)
		yTr.append(current_class)
	current_class+=1


with open ('Y.p','w') as f:
	np.save(f,yTr)

with open('X.p','w') as f:
	np.save(f,xTr)

with open ('Y.p','r') as f:
	trY = np.load(f)
with open('X.p','r') as f:
	trX = np.load(f).reshape(-1,3,40,40)

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
n_examples= n_examples-test_length

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

def predict(model, x_val):
    x = Variable(x_val, requires_grad=False)
    output = model.forward(x)
    return output.data.numpy().argmax(axis=1)

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
print(n_examples)
loss = torch.nn.CrossEntropyLoss(size_average=True)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.)
batch_size = 50
s=time.time()

#print trX
#print trY

for i in range(5):
	cost = 0.
	num_batches = n_examples // batch_size
	for k in range(num_batches):
		start, end = k * batch_size, (k + 1) * batch_size
		if end > n_examples:
			end=n_examples
		#print start,end,n_examples,num_batches
		cost += train(model, loss, optimizer,
				trX[start:end], trY[start:end])
	print("Epoch %d, cost = %f"
		      % (i + 1, cost / num_batches))
e=time.time()
print("time =",e-s)
res = predict(model,teX).reshape(-1,1)
teY=teY.numpy()
acc=0
for i in range(test_length):
	if res[i]==teY[i]:
		acc+=1


acc=acc*1.0/(test_length)
print "accuracy ", acc
print "batch size",batch_size
print "optimizer", "Adam"
print  "parameters", model.parameters
test_example=trX[1:2]
print test_example
model.forward(test_example)
print model.out1[0][0]
print model.out2[0][0]

from matplotlib import pyplot as plt
plt.imshow(model.out1[0][0], interpolation='nearest')
plt.show()