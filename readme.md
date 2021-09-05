# Driver Fatigue Detection | OpenCV
This repo can detect and track user's eyes and mouth and alert when the user is feeling drowsy using video processing techniques.

## Applications
The AAA says that 20% of all fatal accidents in the USA are due to drowsiness! We can only imagine what the stats are like for India which has a higher road accident rate. This system can be used by riders who tend to drive for a longer period of time to warn them and and prevent accidents.

![demo](https://github.com/nimbus1212/driver_drowsiness_detection/blob/master/assets/driver-fatigue-detection.gif)

### Code Requirements
The example code is in Python ([version 2.7](https://www.python.org/download/releases/2.7/) or higher will work). 
The code is tested under Ubuntu 18.04

### Dependencies

- import cv2
- import immutils
- import dlib
- import scipy

### Description
A computer vision system that can automatically detect driver drowsiness by tracking closing of eyes and yawing in a real-time video stream and then alert if the driver appears to be drowsy.

### Execution
To run the code, perform the below command
```
python3 driverFatigue_v2.py
```
or
```
python driverFatigue_v2.py
```
