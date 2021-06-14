import argparse
import cv2
import numpy as np

def hough_circle(image_path, canny):

	# load the image
	image = cv2.imread(image_path)
	if image is None:
		raise TypeError("Image can't be read!")

	if canny:
		edges = cv2.Canny(image, 100, 200)
		cv2.imshow("Canny edge detector", edges)

	# to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# blur to avoid noise
	gray = cv2.medianBlur(gray, 5)

	# compute hough circles
	rows = gray.shape[0] 
	circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 
								minDist=rows/5,
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
		cv2.circle(image, center, radius, (255, 0, 0), thickness=3)

	cv2.imshow("Detected circles", image)
	cv2.waitKey(0)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--image_path", type=str,
                    help="path of the image")
	parser.add_argument("--canny", action="store_true", 
										default=False, 
										help="weather to show edges with Canny detector or not")
	args = parser.parse_args()

	hough_circle(args.image_path,  args.canny)