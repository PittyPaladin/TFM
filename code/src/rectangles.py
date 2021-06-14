import argparse
import cv2
import numpy as np

def find_rectangles(image_path):
	# load the image
	image = cv2.imread(image_path)
	if image is None:
		raise TypeError("Image can't be read!")

	# convert to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cv2.imshow('test', gray)
	cv2.waitKey(0)

	# blur to avoid noise
	# gray = cv2.medianBlur(gray, 5)
	gray = cv2.GaussianBlur(gray, (11,11), 1)
	cv2.imshow('test', gray)
	cv2.waitKey(0)

	# sharpen image
	sharpen_kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
	gray = cv2.filter2D(gray, -1, sharpen_kernel)
	cv2.imshow('test', gray)
	cv2.waitKey(0)

	# sobel edges
	xgrad = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
	ygrad = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
	xgrad = cv2.convertScaleAbs(xgrad)
	ygrad = cv2.convertScaleAbs(ygrad)
	grad = cv2.addWeighted(xgrad, 0.5, ygrad, 0.5, 0)
	cv2.imshow('grad', grad)
	cv2.waitKey(0)
	
	# binarize
	# gray = cv2.adaptiveThreshold(gray,255,
	# 	cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
	# 	cv2.THRESH_BINARY,11,2)
	_, gray = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	cv2.imshow('test', gray)
	cv2.waitKey(0)

	# morphological operators
	kernel = np.ones((7,7), np.uint8)
	gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)
	cv2.imshow('test', gray)
	cv2.waitKey(0)

	# detect contours
	cnts, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# approximate contour and draw convex hull
	hulls = []
	for i in range(len(cnts)):
		epsilon = 0.01*cv2.arcLength(cnts[i], True)
		cnts[i] = cv2.approxPolyDP(cnts[i], epsilon, True)
		hulls.append(cv2.convexHull(cnts[i], False))
	# draw the detected contours
	cv2.drawContours(image, cnts, -1, (0,255,0), 3)
	# draw the convex hull
	cv2.drawContours(image, hulls, -1, (255,0,0), 3)
	cv2.imshow('contours', image)
	cv2.waitKey(0)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--image_path", type=str,
                    help="path of the image")
	args = parser.parse_args()

	find_rectangles(args.image_path)