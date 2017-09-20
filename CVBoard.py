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
im = cv2.imread('Boom.png')
ref = cv2.imread('ocr_a_font.jpg')
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
refCnts = refCnts[0] if imutils.is_cv2() else refCnts[1]
refCnts = contours.sort_contours(refCnts, method='left-to-right')[0]
digits = {}

for(i,c) in enumerate(refCnts):
    (x,y,w,h) = cv2.boundingRect(c)
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

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
locs = []


for(i, c) in enumerate(cnts):
    (x,y,w,h) = cv2.boundingRect(c)
    ar = w / float(h)
    if ar > 2.5 and ar < 4.0:
        if(w > 40 and w < 55) and (h > 10 and h < 20):
            locs.append((x,y,w,h))
            
            
locs = sorted(locs, key=lambda x:x[0])
output = []

imshow(thresh)