import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from pyefd import *
from sklearn.neighbors import KNeighborsClassifier
from skimage import data, io, filters
from skimage import measure
from io import BytesIO

# http://pyefd.readthedocs.io/en/latest/
# Returns Fourier Descriptor Coefficients
def efd_feature(contour):
    coeffs = elliptic_fourier_descriptors(contour, order=20, normalize=True)
    return coeffs
def img_contour(image):
		# Reads image using OpenCV and grayscales
		im = cv2.imread(image)
		imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
		# Soble edge detection using skimage
		edges = filters.sobel(imgray)
		# Find countours from the edges
		countours = measure.find_contours(edges, 0)
		# Extract the coefficients
		return np.asarray(efd_feature(countours[0])).flatten()

# #===============================================================================
# # Run this once to generate csv data to read for future executions
# #===============================================================================
# xTr=[]
# yTr=[]
# current_class=0
# FILEPATH='re_FishImages/combine_image/'
# for files in os.listdir(FILEPATH):
# 	print files
# 	if files == ".DS_Store":
# 		continue
# 	for image in os.listdir(FILEPATH+files):
# 		# Stores the label in column 1 and then the rest of the data to a csv file
# 		xTr.append(np.append(current_class, img_contour(FILEPATH+files+'/'+image)))
# 	current_class+=1
# np.savetxt("pixel_data.csv", xTr, delimiter=",") 
# np.savetxt("rand_data.csv", np.random.permutation(xTr), delimiter=",")
# #===============================================================================

#===============================================================================
# Opens and loads the data into training and test sets.
#===============================================================================
test_length	= 2000
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
	neigh = KNeighborsClassifier(n_neighbors=i)
	# Build a decision tree classifier from the training set
	training = neigh.fit(numpy_x_trn, numpy_y_trn)
	test_accuracy[j]  = "{:.3f}".format((training.score(numpy_x_tst, numpy_y_tst)))
	j+=1

# Plot the data
plt.figure("k-Nearest Neighbors")
plt.plot(k_NN, test_accuracy, color="red", label="Unnormalized")
plt.ylabel('Accuracy')
plt.xlabel('K Neighbor')
plt.legend(loc=4, borderaxespad=0.)
plt.grid()
plt.show()
