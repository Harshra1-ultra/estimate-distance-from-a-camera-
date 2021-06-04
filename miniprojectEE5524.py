"""
Created on Thu Apr  8 14:04:52 2021

@author: harshraj
"""
import cv2
import numpy as np 
img = cv2.imread("/Users/harshraj/IMG_2384.jpg",cv2.IMREAD_GRAYSCALE)
cap=cv2.VideoCapture(0)

# features matrix

sift =cv2.xfeatures2d.SIFT_create()
kp_image, desc_image =sift.detectAndCompute(img, None)
img = cv2.drawKeypoints(img , kp_image, img)
#feature matching 

index_params = dict(algorithm=0, trees=5)
search_params=dict()
flann=cv2.FlannBasedMatcher(index_params, search_params)

while True:
    _,frame = cap.read()
    grayframe = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)   # train image 
    kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)
    matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
    good_points = []
    for m, n in matches :
        if m.distance < 0.54 *n.distance:
            good_points.append(m)
    
    matchimg=cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points, grayframe)
    
    cv2.imshow("maching feature " , matchimg)

    key = cv2.waitKey(1)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
    
    
