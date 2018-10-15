import cv2
import numpy as np
import matplotlib.pyplot as plt


def line_coords(image, line_param):
	slope, intercept = line_param	
	y1 = image.shape[0]
	y2 = int(y1 * (3/5))
	x1 = int((y1 - intercept)/slope)
	x2 = int((y2 - intercept)/slope)
	return np.array([x1, y1, x2, y2])

def av_slope_int(image, lines):
	# Making separate lists for lines on left and right

	left_fit = [] 					
	right_fit = []
	for line in lines:
		x1, y1, x2, y2 = line.reshape(4)
		params = np.polyfit((x1,x2),(y1,y2),1)
		slope = params[0] 
		intcpt = params[1]
		if slope<0:
			left_fit.append((slope, intcpt))
		else:
			right_fit.append((slope, intcpt))
	left_fit_av = np.average(left_fit, axis = 0)
	right_fit_av = np.average(right_fit, axis = 0)
	
	left_line = line_coords(image, left_fit_av)
	right_line = line_coords(image, right_fit_av)
	return np.array([left_line, right_line])

def canny(image):
	# Grayscale Conversion
	gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
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
			# print(line)
			x1, y1, x2, y2 = line.reshape(4)
			cv2.line(line_image, (x1,y1), (x2,y2),(255,0,0), 10)
	return line_image


plt.close("all")

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)

# Canny Edge Detection

canny_im = canny(lane_image)


# Observing a specified region 
# in the image, formed by: [200,0], [550,200], [1000,0], [200,0]

masked_image = reg_interest(canny_im )

#  Extracting Lines from an image using Hough Transform
#  Plotting the extracted lines on the black background
lines = cv2.HoughLinesP(masked_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
line_image = disp_lines(lane_image, lines)

# Obtaining the average / mean of the detected line on the image

av_lines = av_slope_int(lane_image, lines)

# Plotting the averaged line on the lane image

line_image = disp_lines(lane_image, av_lines)

# Plotting the extracted lines on the original image

comb_im = cv2.addWeighted(lane_image, 0.99, line_image, 1, 1)

cv2.imshow('result',comb_im)
cv2.waitKey(0)