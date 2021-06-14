import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
	# query image
	img = cv2.imread("../img/cocacola_database.png")
	imGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# scene image
	scene = cv2.imread("../img/cocacola0_pack.jpg")
	scGray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

	# find the keypoints and descriptors with SIFT
	sift = cv2.SIFT_create()
	imKp, imDes = sift.detectAndCompute(imGray, None)
	scKp, scDes = sift.detectAndCompute(scGray, None)

	# draw the keypoints for the query image
	kpsImg = cv2.drawKeypoints(imGray, imKp, img, 
		flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)	
	cv2.imshow("keypoints in image", kpsImg)
	cv2.waitKey(0)

	# use FLANN for knn-based matching
	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)   # or pass empty dictionary
	flann = cv2.FlannBasedMatcher(index_params,search_params)
	matches = flann.knnMatch(imDes, scDes, k=2)

	# Need to draw only good matches, so create a mask
	matchesMask = [[0,0] for i in range(len(matches))]

	# ratio test as per Lowe's paper
	goodMatches = []
	for m, n in matches:
		if m.distance < 0.7*n.distance:
			goodMatches.append(m)

	min_matches = 10
	if len(goodMatches) >= min_matches:
		src_pts = np.float32([ imKp[m.queryIdx].pt for m in goodMatches ]).reshape(-1,1,2)
		dst_pts = np.float32([ scKp[m.trainIdx].pt for m in goodMatches ]).reshape(-1,1,2)

		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		matchesMask = mask.ravel().tolist()

		h, w = imGray.shape
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		dst = cv2.perspectiveTransform(pts,M)

		scGray = cv2.polylines(scGray, [np.int32(dst)], True, 0, 3, cv2.LINE_AA)
	
	else:
		print( "Not enough matches are found - {}/{}".format(len(goodMatches), min_matches) )
		matchesMask = None
	
	draw_params = dict(
		matchColor = (0,255,0), # draw matches in green color
		singlePointColor = None,
		matchesMask = matchesMask, # draw only inliers
		flags = 2)
	
	matchesDrawn = cv2.drawMatches(imGray, imKp, scGray, scKp, goodMatches, None, **draw_params)
	
	plt.imshow(matchesDrawn)
	plt.show()