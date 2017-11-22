import cv2
import numpy as np
from pyefd import *
from skimage import data, io, filters
from skimage import measure

# http://pyefd.readthedocs.io/en/latest/
# Returns Fourier Descriptor Coefficients
def efd_feature(contour):
    coeffs = elliptic_fourier_descriptors(contour, order=20, normalize=True)
    return coeffs

# Reads image using OpenCV and grayscales
im = cv2.imread('fish_000000249596_03656.png')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

# Soble edge detection using skimage
edges = filters.sobel(imgray)
# Find countours from the edges
countours = measure.find_contours(edges, 0)
# Extract the coefficients
coeffs = efd_feature(countours[0])

# Plots Fourier Descriptor
plot_efd(coeffs)
# Plots edges
io.imshow(edges)
io.show()
