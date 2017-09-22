# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:44:01 2017

@author: Nguyen
"""

#Linear Algebra
import numpy as np

#Visualization
import matplotlib.pyplot as plt

#Images
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.color import rgb2gray
from imutils import contours
import argparse
import imutils
import cv2

#CV to read in card and get data

#Read in card image
im = cv2.imread('boom.png')
ref = cv2.imread('fonttry.png')
orig = ref
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

contours_im = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
refCnts = contours_im[0] if imutils.is_cv2() else contours_im[1]
refCnts = contours.sort_contours(refCnts, method='left-to-right')[0]
digits = {}
holder = 0
for(i,c) in enumerate(refCnts):
    (x,y,w,h) = cv2.boundingRect(c)
    
    cv2.rectangle(orig, (x,y),(x+w,y+h),(0,0,255),2)
    roi = ref[y:y+h, x:x+w]
    roi = cv2.resize(roi, (57,88))
    
    digits[i] = roi
          
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))


im = imutils.resize(im, width = 300)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
gradX = np.absolute(gradX)

#Minmax normalize
(minVal,maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX-minVal)/ (maxVal-minVal)))
gradX = gradX.astype('uint8')

gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
locs = []

for(i, c) in enumerate(cnts):
    (x,y,w,h) = cv2.boundingRect(c)
    ar = w / float(h)
    if ar > 0 and ar < 20.0:
        if(w > 50 and w < 1000) and (h > 20 and h < 35):#if(w > 60 and w < 1000) and (h > 30 and h < 35):
            cv2.rectangle(im, (x,y), (x+w, y+h), (0,0,255),2)
            locs.append((x,y,w,h))
        
            
locs = sorted(locs, key=lambda x:x[0])
output = []
for(i, (gX,gY,gW,gH)) in enumerate(locs):
    groupOutput = []
    group = gray[gY+3:gY + gH + 3, gX-3:gX + gW + 10]
    o = cv2.cvtColor(group, cv2.COLOR_GRAY2RGB)
    #cv2.rectangle(o, (x,y),(x+w,y+h),(0,255,0),2)
    group = cv2.threshold(group, 0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    digitCnts = cv2.findContours(group.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = digitCnts[0] if imutils.is_cv2() else digitCnts[1]
    digitCnts = contours.sort_contours(digitCnts, method = "left-to-right")[0]
    for c in digitCnts:
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(o, (x,y),(x+w,y+h),(0,0,255),2)
        #if(w > 5 and w < 50) and (h > 5 and h < 20):
        
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57,88))#(57,88))
        scores = []
        for(digit, digitROI) in digits.items():
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            (_, score, _ , _) = cv2.minMaxLoc(result)
            scores.append(score)
        groupOutput.append(str(np.argmax(scores)))
            
    

#print(refCnts[0], refCnts[0].shape, refCnts[0][2])
#print(len(contours_im))
#print(np.array(counts).shape)
#locs = np.array(locs).reshape((-1,1,2)).astype(np.int32)
#print(cnts[28])
print(len(group))
print(groupOutput)
cv2.imshow("asdf", digits[38])

#cardName = ""
#for i in range(0,len(groupOutput)):
#    cardName += str(chr(int(groupOutput[i])+97))
    
#print(cardName)

#cv2.drawContours(im,locs,-1,(255,0,0),20) #32 is name
#imshow(orig)
#imshow(orig)

#imshow(ref)