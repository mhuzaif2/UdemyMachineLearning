import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
	# Grayscale Conversion
	gray = cv2.cvtColor(lane_image,cv2.COLOR_RGB2GRAY)
	#  Gaussian Filter
	blur_g = cv2.GaussianBlur(gray, (3,3), 0)
	blur_simple = cv2.blur(gray, (3,3), 0)
	#  Canny Method to identify Edges
	canny_g = cv2.Canny(blur_g,50,150)
	canny_s = cv2.Canny(blur_simple,50,150)
	return(canny_g)

def reg_interest(image):
	h = image.shape[0]
	pgns = np.array([[(200,h),(1100, h), (550, 250)]])
	mask = np.zeros_like(image)
	cv2.fillPoly(mask, pgns, 255)
	return mask


image = cv2.imread('test_image.jpg')

lane_image = np.copy(image)


canny = canny(lane_image)

cv2.imshow('result',reg_interest(canny))
cv2.waitKey(0)
# plt.imshow(lane_image)
# plt.show()

# Observing a specified region 
# in the image, formed by: [200,0], [550,200], [1000,0], [200,0]

