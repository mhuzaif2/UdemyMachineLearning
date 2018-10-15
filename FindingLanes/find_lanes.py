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
	masked_image = cv2.bitwise_and(image, mask)
	return masked_image

def disp_lines(image, lines):
	line_image = np.zeros_like(image)
	if lines is not None:
		for line in lines:
			print(line)
			x1, y1, x2, y2 = line.reshape(4)
			cv2.line(line_image, (x1,y1), (x2,y2),(255,0,0), 10)
	return line_image


image = cv2.imread('test_image.jpg')

lane_image = np.copy(image)


canny = canny(lane_image)

# cv2.imshow('result',reg_interest(canny))
# cv2.waitKey(0)

# Observing a specified region 
# in the image, formed by: [200,0], [550,200], [1000,0], [200,0]

masked_image = reg_interest(canny)

#  Extracting Lines from an image using Hough Transform
#  Plotting the extracted lines on the black background
lines = cv2.HoughLinesP(masked_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
line_image = disp_lines(lane_image, lines)

# Plotting the extracted lines on the original image

comb_im = cv2.addWeighted(lane_image, 0.6, line_image, 1, 1)

cv2.imshow('result',comb_im)
cv2.waitKey(0)