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

	

image = cv2.imread('test_image.jpg')

lane_image = np.copy(image)


canny = canny(lane_image)

#  first argument the name of the window, 
# second is the image record
# cv2.imshow('Edge_g',canny) 
# cv2.waitKey(0)

plt.imshow(lane_image)
plt.show()


