#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 18:10:29 2021
@author: harshraj
"""

import cv2
import numpy as np 
img = cv2.imread("/Users/harshraj/IMG_2384.jpg",cv2.IMREAD_GRAYSCALE)
cap=cv2.VideoCapture(0)

# features matrix

sift =cv2.xfeatures2d.SIFT_create()
kp_image, desc_image =sift.detectAndCompute(img, None)
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
        if m.distance < 0.5 *n.distance:
            good_points.append(m)
    
    # matchimg=cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points, grayframe)
    
    #Homogrophy 
    if len(good_points)>30:
        query_pts=np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts=np.float32([kp_image[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        
        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        #prespective transform
        h,w = img.shape
        pts =np.float32([[0,0],[h,0],[0,w],[h,w]]).reshape(-1,1,2)
        dst= cv2.perspectiveTransform(pts, matrix)
        
        homogrophy=cv2.polylines(frame ,[np.int32(dst)], True , (0,255,0),3)
        cv2.imshow("Homography", homogrophy)
    else:
        cv2.imshow("Homograpy", grayframe)
        
    #cv2.imshow("grayframe ", grayframe)
    #cv2.imshow("maching feature " , matchimg)

    key = cv2.waitKey(1)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
    
    