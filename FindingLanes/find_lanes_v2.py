import cv2
import numpy as np
import matplotlib.pyplot as plt


def line_coords(image, line_param):
	slope, intercept = line_param	
	y1 = image.shape[0]
	y2 = int(y1 * (3/5))
	x1 = int((y1 - intercept)/slope)
	x2 = int((y2 - intercept)/slope)
	return [[x1, y1, x2, y2]]

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
	if len(left_fit) and len(right_fit):
		left_fit_av = np.average(left_fit, axis = 0)
		right_fit_av = np.average(right_fit, axis = 0)
		left_line = line_coords(image, left_fit_av)
		right_line = line_coords(image, right_fit_av)
		return [left_line, right_line]

def canny(image):
	# Grayscale Conversion
	gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
	#  Gaussian Filter
	kernel = 5;
	blur_g = cv2.GaussianBlur(gray, (kernel,kernel), 0)
	# blur_simple = cv2.blur(gray, (3,3), 0)
	#  Canny Method to identify Edges
	canny_g = cv2.Canny(blur_g,50,150)
	# canny_s = cv2.Canny(blur_simple,50,150)
	return(canny_g)

def reg_interest(canny_image):
	h = canny_image.shape[0]
	pgns = np.array([[(200,h), (550, 250),(1100, h)]],np.int32)
	mask = np.zeros_like(canny_image)
	cv2.fillPoly(mask, pgns, 255)
	masked_image = cv2.bitwise_and(canny_image, mask)
	return masked_image

def disp_lines(image, lines):
	line_image = np.zeros_like(image)
	if lines is not None:
		for line in lines:			
			for x1, y1, x2, y2 in line:			 
				cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)
	return line_image



cap = cv2.VideoCapture('test2.mp4')
while(cap.isOpened()):
	ret, frame = cap.read()
	if ret==True:
		canny_image = canny(frame)

		# Observing a specified region 
		# in the image, formed by: [200,0], [550,200], [1000,0], [200,0]

		masked_image = reg_interest(canny_image )

		#  Extracting Lines from an image using Hough Transform
		#  Plotting the extracted lines on the black background
		lines = cv2.HoughLinesP(masked_image, 5, np.pi/180, 100, np.array([]), minLineLength = 10, maxLineGap = 5)	

		# Obtaining the average / mean of the detected line on the image

		av_lines = av_slope_int(frame, lines)

		# Plotting the averaged line on the lane image

		line_image = disp_lines(frame, av_lines)

		# Plotting the extracted lines on the original image

		comb_im = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

		cv2.imshow('result',comb_im)
		if cv2.waitKey(1) == ord('q'):
			break
	else:
		break
cap.release()
cv2.destroyAllWindows()