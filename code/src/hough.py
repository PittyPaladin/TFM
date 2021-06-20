import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

def hough_circle(image_path, canny):

	# load the image
	image = cv2.imread(image_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	if image is None:
		raise TypeError("Image can't be read!")

	if canny:
		edges = cv2.Canny(image, 100, 200)

	# to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# blur to avoid noise
	gray = cv2.medianBlur(gray, 5)

	# compute hough circles
	rows = gray.shape[0] 
	circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 
								minDist=rows/50,
								param1=100, param2=100,
								minRadius=10, maxRadius=200)
	# NOTE: minimum distance between circle centers is proportional to the 
	# dimensions of the image
	if circles is None:
		print("No circles were detected!")
		return
	circles = np.squeeze(circles) # take out extra dimension
	for circle in circles:
		center = (int(circle[0]), int(circle[1]))
		radius = int(circle[2])
		cv2.circle(image, center, radius, (0, 255, 0), thickness=3)

	fig, axs = plt.subplots(nrows=2, ncols=1)
	axs.ravel()
	axs[0].imshow(image)
	axs[0].set_axis_off()
	axs[1].imshow(edges)
	axs[1].set_axis_off()
	plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--image_path", type=str,
                    help="path of the image")
	parser.add_argument("--canny", action="store_true", 
										default=True, 
										help="weather to show edges with Canny detector or not")
	args = parser.parse_args()

	hough_circle(args.image_path,  args.canny)