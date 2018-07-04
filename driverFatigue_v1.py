import cv2
import numpy as np
import time
import sys

path = "classifiers/haar-face.xml"
faceCascade = cv2.CascadeClassifier(path)


# Variable used to hold the ratio of the contour area to the ROI
ratio = 0

# variable used to hold the average time duration of the yawn
global yawnStartTime
yawnStartTime = 0

# Flag for testing the start time of the yawn
global isFirstTime
isFirstTime = True

# List to hold yawn ratio count and timestamp
yawnRatioCount = []

# Yawn Counter
yawnCounter = 0

# yawn time
averageYawnTime = 2.5

"""
Find the second largest contour in the ROI;
Largest is the contour of the bottom half of the face.
Second largest is the lips and mouth when yawning.
"""
def calculateContours(image, contours):
	cv2.drawContours(image, contours, -1, (0,255,0), 3)
	maxArea = 0
	secondMax = 0
	maxCount = 0
	secondmaxCount = 0
	for i in contours:
		count = i
		area = cv2.contourArea(count)
		if maxArea < area:
			secondMax = maxArea
			maxArea = area
			secondmaxCount = maxCount
			maxCount = count
		elif (secondMax < area):
			secondMax = area
			secondmaxCount = count

	return [secondmaxCount, secondMax]

"""
Thresholds the image and converts it to binary
"""
def thresholdContours(mouthRegion, rectArea):
	global ratio

	# Histogram equalize the image after converting the image from one color space to another
	# Here, converted to greyscale
	imgray = cv2.equalizeHist(cv2.cvtColor(mouthRegion, cv2.COLOR_BGR2GRAY))

	# Thresholding the image => outputs a binary image.
	# Convert each pixel to 255 if that pixel each exceeds 64. Else convert it to 0.
	ret,thresh = cv2.threshold(imgray, 64, 255, cv2.THRESH_BINARY)

	# Finds contours in a binary image
	# Constructs a tree like structure to hold the contours
	# Contouring is done by having the contoured region made by of small rectangles and storing only the end points
	# of the rectangle
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	returnValue = calculateContours(mouthRegion, contours)

	# returnValue[0] => secondMaxCount
	# returnValue[1] => Area of the contoured region.
	secondMaxCount = returnValue[0]
	contourArea = returnValue[1]

	ratio = contourArea / rectArea

	# Draw contours in the image passed. The contours are stored as vectors in the array.
	# -1 indicates the thickness of the contours. Change if needed.
	if(isinstance(secondMaxCount, np.ndarray) and len(secondMaxCount) > 0):
		cv2.drawContours(mouthRegion, [secondMaxCount], 0, (255,0,0), -1)

"""
Isolates the region of interest and detects if a yawn has occured.
"""
def yawnDetector(video_capture):
	global ratio, yawnStartTime, isFirstTime, yawnRatioCount, yawnCounter

   	# Capture frame-by-frame
	ret, frame = video_capture.read()

	gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
	gray = cv2.GaussianBlur(gray, (5, 5), 0)

	faces = faceCascade.detectMultiScale(
        	gray,
        	scaleFactor=1.1,
        	minNeighbors=5,
        	minSize=(50, 50),
        	flags=cv2.CV_HAAR_SCALE_IMAGE
    	)

	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

		# Isolate the ROI as the mouth region
		widthOneCorner = (x + (w / 4))
		widthOtherCorner = x + ((3 * w) / 4)
		heightOneCorner = y + (11 * h / 16)
		heightOtherCorner = y + h

		# Indicate the region of interest as the mouth by highlighting it in the window.
		cv2.rectangle(frame, (widthOneCorner, heightOneCorner), (widthOtherCorner, heightOtherCorner),(0,0,255), 2)

		# mouth region
		mouthRegion = frame[heightOneCorner:heightOtherCorner, widthOneCorner:widthOtherCorner]

		# Area of the bottom half of the face rectangle
		rectArea = (w*h)/2

		if(len(mouthRegion) > 0):
			thresholdContours(mouthRegion, rectArea)

		print ("Current probablity of yawn: " + str(round(ratio*1000, 2)) + "%")
		print ("Length of yawnCounter: " + str(len(yawnRatioCount)))

		if(ratio > 0.06):
			if(isFirstTime is True):
				isFirstTime = False
				yawnStartTime = time.time()

			# If the mouth is open for more than 2.5 seconds, classify it as a yawn
			if((time.time() - yawnStartTime) >= averageYawnTime):
				yawnCounter += 1
				yawnRatioCount.append(yawnCounter)

				if(len(yawnRatioCount) > 8):
					# Reset all variables
					isFirstTime = True
					yawnStartTime = 0
					return True


	# Display the resulting frame
	cv2.namedWindow('yawnVideo')
	cv2.imshow('yawnVideo', frame)
	time.sleep(0.025)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		sys.exit(0)

	return False

"""
Main
"""
def main():
	# Capture from web camera
	yawnCamera = cv2.VideoCapture(0)

	while True:
		returnValue = (yawnDetector(yawnCamera), 'yawn')
		if returnValue[0]:
			print ("Yawn detected!")
			# When everything is done, release the capture
			yawnCamera.release()
			cv2.destroyWindow('yawnVideo')
			return returnValue


main()
