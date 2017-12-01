import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import misc
from pyefd import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from skimage import data, io, filters
from skimage import measure
from scipy.stats import skew
from io import BytesIO

# http://pyefd.readthedocs.io/en/latest/
# Returns Fourier Descriptor Coefficients
def efd_feature(contour):
    coeffs = elliptic_fourier_descriptors(contour, order=8, normalize=True)
    return coeffs
def img_contour(image):
		# Reads image using OpenCV and grayscales
		im = cv2.imread(image)
		imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

		mask = np.zeros(imgray.shape,np.uint8)

		# Soble edge detection using skimage
		edges = filters.sobel(imgray)
		# Find countours from the edges
		countours = measure.find_contours(edges, 0)
		# Extract the coefficients
		coeff = np.asarray(efd_feature(countours[0])).flatten()
		# Append meanStd and skewness of color
		meanStdSkew = np.append(np.asarray(cv2.meanStdDev(im)).flatten(), skew(im[:,:,0], axis=None))
		meanStdSkew = np.append(meanStdSkew, skew(im[:,:,1], axis=None))
		meanStdSkew = np.append(meanStdSkew, skew(im[:,:,2], axis=None))
		return meanStdSkew

#===============================================================================
# Run this once to generate csv data to read for future executions
#===============================================================================
xTr=[]
yTr=[]
current_class=0
FILEPATH='re_FishImages/combine_image/'
for files in os.listdir(FILEPATH):
	print files
	if files == ".DS_Store":
		continue
	for image in os.listdir(FILEPATH+files):
		data = np.append(current_class, img_contour(FILEPATH+files+'/'+image))
		# Stores the label in column 1 and then the rest of the data to a csv file
		xTr.append(data)
	current_class+=1
np.savetxt("pixel_data.csv", xTr, delimiter=",")
# Randomizes the data which we will use for our machine learning 
np.savetxt("rand_data.csv", np.random.permutation(xTr), delimiter=",")
#===============================================================================

#===============================================================================
# Opens and loads the data into training and test sets.
#===============================================================================
test_length	= 4000
with open('./rand_data.csv', "rb") as input_file:
	x_val  = input_file.read()
numpy_x_val = np.genfromtxt(BytesIO(x_val), delimiter=",", dtype=float)
numpy_y_val = numpy_x_val[:,0]
numpy_x_val = numpy_x_val[:,1:]

numpy_x_tst = numpy_x_val[0:test_length,:]
numpy_y_tst = numpy_y_val[0:test_length]

numpy_x_trn = numpy_x_val[test_length:,:]
numpy_y_trn = numpy_y_val[test_length:]

#===============================================================================
# K-Nearest Neightbor Classifier
#===============================================================================
k_NN					 = [1,3,5,7,9,11,13,15]
test_accuracy  = [0] * 8
j = 0
for i in k_NN:
	# Use scikit to create the tree for us
	training = KNeighborsClassifier(n_neighbors=i)
	# Build a decision tree classifier from the training set
	training = training.fit(numpy_x_trn, numpy_y_trn)
	test_accuracy[j]  = "{:.3f}".format((training.score(numpy_x_tst, numpy_y_tst)))
	j+=1

print("neighbors \t test")
j = 0
for i in k_NN:
	print "%f \t %f" %(float(i),float(test_accuracy[j]))
	j+=1

# Plot the data
plt.figure("k-Nearest Neighbors")
plt.plot(k_NN, test_accuracy, color="red", label="Test Set")
plt.ylabel('Accuracy')
plt.xlabel('K Neighbor')
plt.grid()

#===============================================================================
# SVM Classifier
#===============================================================================

test_accuracy0  = [0] * 9
svm_c     		  = [0.0001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
log_svm 			  = [0] * 9

j = 0
for i in svm_c:
	log_svm[j]  = math.log10(svm_c[j])
	j+=1

j = 0
for i in svm_c:
	# Use scikit to create the tree for us
	training = SVC(C=i, kernel='rbf')
	# Build a decision tree classifier from the training set
	training = training.fit(numpy_x_trn, numpy_y_trn)
	test_accuracy0[j]  = "{:.3f}".format((training.score(numpy_x_tst, numpy_y_tst)))
	j+=1

print("C value \t test")
j = 0
for i in svm_c:
	print "%f \t %f" %(float(i),float(test_accuracy0[j]))
	j+=1

plt.figure("SVM-linear")
plt.plot(log_svm, test_accuracy0, color="red", label="Test Set")
plt.ylabel('Accuracy')
plt.xlabel('Penalty Parameter(log)')
plt.grid()

#===============================================================================
# Logistic Regression
#===============================================================================

# Initialize variables
test_accuracy1   = [0] * 30
regularizer		   = [0] * 30
log_regular	     = [0] * 30

for i in range(30):
	regularizer[i] = 0.0000001 * pow(2,i)
	log_regular[i] = i

j = 0
for i in regularizer:
	# Use scikit to create the tree for us
	training = LogisticRegression(C=regularizer[j], solver='liblinear', max_iter=20000)
	# Build a decision tree classifier from the training set
	training = training.fit(numpy_x_trn, numpy_y_trn)
	test_accuracy1[j]   = "{:.3f}".format((training.score(numpy_x_tst, numpy_y_tst)))
	j+=1

j = 0
print("\nLogistic Regression: ")
print("Regularization \t test")
for i in regularizer:
	print "%f \t %f" %(float(i),float(test_accuracy1[j]))
	j+=1

plt.figure("Logistic Regression")
plt.plot(log_regular, test_accuracy1, color="red", label="Test Set")
plt.ylabel('Accuracy')
plt.xlabel('Regularization (log)')
plt.grid()

plt.show()
