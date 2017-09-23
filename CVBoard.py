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
from skimage import exposure
from skimage.filters import rank
from skimage.filters.rank import autolevel
from skimage.filters.rank import median
from skimage.filters.rank import mean_bilateral
from skimage.morphology import disk
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.color import rgb2gray
from imutils import contours
import argparse
import imutils
import math
import cv2

from PIL import Image
#CV to read in card and get data

#Read in card image
im = cv2.imread('ui.png')
ref = cv2.imread('lower.png')
orig = ref
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
cv2.bitwise_not(ref)
ref = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

contours_im = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
refCnts = contours_im[0] if imutils.is_cv2() else contours_im[1]
refCnts = contours.sort_contours(refCnts, method='left-to-right')[0]
digits = {}
holder = 0
for(i,c) in enumerate(refCnts):
    (x,y,w,h) = cv2.boundingRect(c)
    
    cv2.rectangle(orig, (x,y),(x+w,y+h),(0,0,255),2)
    roi = ref[y:y+h+ 1, x:x+w+ 1]
    roi = cv2.resize(roi, (57,88))
    roi = rank.equalize(roi, disk(5))
    digits[i] = roi
          
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))


im = imutils.resize(im, width = 300)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.bitwise_not(gray)
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
gradX = np.absolute(gradX)

#Minmax normalize
(minVal,maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX-minVal)/ (maxVal-minVal)))
gradX = gradX.astype('uint8')

gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
locs = []

for(i, c) in enumerate(cnts):
    (x,y,w,h) = cv2.boundingRect(c)
    ar = w / float(h)
    if ar > 0 and ar < 20.0:
        if(w > 60 and w < 1000) and (h > 20 and h < 35):#if(w > 60 and w < 1000) and (h > 30 and h < 35):
            locs.append((x,y,w,h))
        
            
locs = sorted(locs, key=lambda x:x[0])
output = []
crops = {}
non_rotated = {}
allout = []
kernel = np.ones((3,3), np.uint8)
idx = 0
for(i, (gX,gY,gW,gH)) in enumerate(locs):
    groupOutput = []
    group = gray[gY - 1:gY + gH, gX:gX + gW]
    o = cv2.cvtColor(group, cv2.COLOR_GRAY2RGB)
    group = cv2.threshold(group,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #digitCnts = cv2.findContours(group.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #digitCnts = digitCnts[0] if imutils.is_cv2() else digitCnts[1]
    #digitCnts = contours.sort_contours(digitCnts, method = "left-to-right")[0]
    cv2.drawContours(o, digitCnts, -1, (0,255,0), 1)
    
    for c in digitCnts:
        (x,y,w,h) = cv2.boundingRect(c)
        #ar = w / float(h)
        #if ar > .4 and ar < 3.0:
        if(w > 3 and w < 20) and (h > 5 and h < 20):
            
            cv2.rectangle(o, (x,y),(x+w,y+h),(0,0,255),1)
            roi = group[y :y+ h + 2, x:x + w]
            non_rotated[idx] = cv2.resize(roi, (57,88))
            coords = np.column_stack(np.where(roi > 0))
            angle = cv2.minAreaRect(coords)[-1]
            if(angle < -45):
                angle = -(90 + angle)
            else:
                angle = -angle
            (h , w) = roi.shape[:2]
            center = (w//2, h//2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(roi, M, (w, h), flags = cv2.INTER_CUBIC, borderMode = cv2.BORDER_REPLICATE)
            rotated = rank.equalize(rotated, disk(3))
            cv2.rectangle(o, (x,y),(x+w,y+h),(0,0,255),1)
            roi = cv2.resize(rotated, (57,88))#(57,88))
            crops[idx] = roi
            idx += 1
            scores = []
            for(digit, digitROI) in digits.items():
                result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
                #(_, score, _ , _) = cv2.minMaxLoc(result)
                scores.append(result)
            allout.append(scores)
            groupOutput.append(str(np.argmax(scores)))       
    

#print(refCnts[0], refCnts[0].shape, refCnts[0][2])
#print(len(contours_im))
#print(np.array(counts).shape)
#locs = np.array(locs).reshape((-1,1,2)).astype(np.int32)
#print(cnts[28])
print(allout[0])
print(groupOutput)
outDict = {20: 'u', 5: 'f', 19: 't', 8: 'i', 12:'m',0:'a',4:'e', 11:'l'}
#cv2.imshow("daf", o)
cv2.imshow("asdf", o)
cv2.imshow("CHAR", crops[9])
cv2.imshow("actual", digits[0])
cv2.imshow("template", orig)
#cv2.imshow("stuff", non_rotated[1])


#cv2.drawContours(im,locs,-1,(255,0,0),20) #32 is name
#imshow(orig)
#imshow(orig)

#imshow(ref)