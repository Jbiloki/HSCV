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
from skimage.filters import rank
from skimage.morphology import disk
from imutils import contours
import imutils
import cv2
import difflib
import pandas as pd

groupOutput = []
ref = cv2.imread('lighter copy.png')
db = pd.read_json('cards.json')
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
cv2.bitwise_not(ref)
ref = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

contours_im = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
refCnts = contours_im[0] if imutils.is_cv2() else contours_im[1]
refCnts = contours.sort_contours(refCnts, method='left-to-right')[0]
digits = {}
for(i,c) in enumerate(refCnts):
    (x,y,w,h) = cv2.boundingRect(c)
    
    #cv2.rectangle(orig, (x,y),(x+w,y+h),(0,0,255),2)
    roi = ref[y:y+h+ 1, x :x+w]
    roi = cv2.resize(roi, (57,88))
    roi = rank.equalize(roi, disk(10))
    digits[i] = roi
#CV to read in card and get data
def addFromBoard(im,locs, gray):
    #Read in card image
    cardName = ""
    locs = sorted(locs, key=lambda x:x[0])
    crops = {}
    non_rotated = {}
    allout = []
    idx = 0
    for(i, (gX,gY,gW,gH)) in enumerate(locs):
        global groupOutput
        group = gray[gY - 5:gY + gH, gX:gX + gW]
        cv2.imshow("group", group)
        #o = cv2.cvtColor(group, cv2.COLOR_GRAY2RGB)
        group = cv2.threshold(group,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        digitCnts = cv2.findContours(group.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digitCnts = digitCnts[0] if imutils.is_cv2() else digitCnts[1]
        digitCnts = contours.sort_contours(digitCnts, method = "left-to-right")[0]
        #cv2.drawContours(o, digitCnts, -1, (0,255,0), 1)
        
        for c in digitCnts:
            (x,y,w,h) = cv2.boundingRect(c)
            if(w > 3 and w < 20) and (h > 5 and h < 20):
                
                #cv2.rectangle(o, (x,y),(x+w,y+h),(0,0,255),1)
                roi = group[y - 1:y+ h + 2, x:x + w]
                if roi.any():
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
                    #cv2.rectangle(o, (x,y),(x+w,y+h),(0,0,255),1)
                    roi = cv2.resize(rotated, (57,88))#(57,88))
                    crops[idx] = roi
                    idx += 1
                    scores = []
                    for(digit, digitROI) in digits.items():
                        result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
                        (_, score, _ , _) = cv2.minMaxLoc(result)
                        scores.append(result)
                    allout.append(scores)
                    groupOutput.append(str(np.argmax(scores)))       
    
    
    #print(refCnts[0], refCnts[0].shape, refCnts[0][2])
    #print(len(contours_im))
    #print(np.array(counts).shape)
    #locs = np.array(locs).reshape((-1,1,2)).astype(np.int32)
    #print(cnts[28])
    #print(allout[0])
    outDict = {0:'A',1:'a',2:'b',3:'B',4:'c',5:'C',6:'d',7:'D',8:'e',9:'E',10:'f',11:'F',12:'g',13:'G',14:'h',15:'H',16:'i',17:'j',18:'I',19:'k',20:'J',21:'l',22:'K',23:'m',
               24:'L',25:'n',26:'M',27:'o',28:'N',29:'p',30:'O',31:'q',32:'p',33:'r',34:'Q',35:'S',36:'t',37:'R',38:'u',39:'S',40:'v',41:'T',42:'w',43:'U',44:'x',45:'V',46:'y',47:'W',48:'z',49:'X',
               50:'Y',51:'Z'}
    #cv2.imshow("daf", o)
    #cv2.imshow("asdf", o)
    #cv2.imshow("true", im)
    #cv2.imshow("template", orig)
    cardName = ""
    print(len(groupOutput))
    
    for i in range(len(groupOutput)):
        cardName += outDict[int(groupOutput[i])]
    print("Card: ", cardName)
    groupOutput = []
    print(difflib.get_close_matches(cardName,db['name']))
    #cv2.drawContours(im,locs,-1,(255,0,0),20) #32 is name
    #imshow(orig)
    #imshow(orig)
    
    #imshow(ref)