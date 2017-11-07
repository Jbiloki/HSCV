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

#Load in our template for font matching
ref = cv2.imread('lighter copy.png')

#Load in card database to for lookup
db = pd.read_json('cards.json')

#Prepare templates
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
cv2.bitwise_not(ref)
ref = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

contours_im = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
refCnts = contours_im[0] if imutils.is_cv2() else contours_im[1]
refCnts = contours.sort_contours(refCnts, method='left-to-right')[0]
digits = {}
for(i,c) in enumerate(refCnts):
    (x,y,w,h) = cv2.boundingRect(c)
    roi = ref[y:y+h+ 1, x :x+w]
    roi = cv2.resize(roi, (57,88))
    roi = rank.equalize(roi, disk(10))
    digits[i] = roi
    

#CV to read in card and get data
def addFromBoard(im,locs, gray):
    #Read in card image
    cardName = ""
    locs = sorted(locs, key=lambda x:x[0])
    crops = {} # Used to check each letter after crop
    idx = 0 # Used to index crops
    for(i, (gX,gY,gW,gH)) in enumerate(locs):
        global groupOutput
        
        #Cut out the group of the name
        group = gray[gY - 5:gY + gH, gX:gX + gW]
        #cv2.imshow("group", group) Used for debugging, see what name or roi is passed here

        #Process group
        group = cv2.threshold(group,10,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        digitCnts = cv2.findContours(group.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digitCnts = digitCnts[0] if imutils.is_cv2() else digitCnts[1]
        digitCnts = contours.sort_contours(digitCnts, method = "left-to-right")[0]
        
        for c in digitCnts:
            (x,y,w,h) = cv2.boundingRect(c)
            #Only process and check for letters of certain roi size to avoid noise
            if(w > 3 and w < 20) and (h > 5 and h < 20):
                roi = group[y - 1:y+ h + 2, x:x + w]
                
                #Make sure the roi is note None
                if roi.any():
                    
                    #Rotate roi due to card skew
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
                    
                    #Resize to match template
                    roi = cv2.resize(rotated, (57,88))#(57,88))
                    crops[idx] = roi
                    idx += 1
                    scores = []
                    #Compare roi to each template and get scores, take max score as letter
                    for(digit, digitROI) in digits.items():
                        result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
                        (_, score, _ , _) = cv2.minMaxLoc(result)
                        scores.append(result)
                    groupOutput.append(str(np.argmax(scores)))       
    
    #Dictonary of our templates to matching letters
    outDict = {0:'A',1:'a',2:'b',3:'B',4:'c',5:'C',6:'d',7:'D',8:'e',9:'E',10:'f',11:'F',12:'g',13:'G',14:'h',15:'H',16:'i',17:'j',18:'I',19:'k',20:'J',21:'l',22:'K',23:'m',
               24:'L',25:'n',26:'M',27:'o',28:'N',29:'p',30:'O',31:'q',32:'p',33:'r',34:'Q',35:'S',36:'t',37:'R',38:'u',39:'S',40:'v',41:'T',42:'w',43:'U',44:'x',45:'V',46:'y',47:'W',48:'z',49:'X',
               50:'Y',51:'Z'}
    
    #Loop through template letters found and get value from dict
    cardName = ""
    for i in range(len(groupOutput)):
        cardName += outDict[int(groupOutput[i])]
    #print("Card: ", cardName)
    groupOutput = []
    #Get closest match from our card database
    if cardName != "":
        currCard = difflib.get_close_matches(cardName,db['name'])
        if currCard != []:
            print("Card added to your hand")
            return {'name': str(db[db['name'] == currCard[0]].name.item()), 'attack':str(db[db['name'] == currCard[0]].attack.item()),'health':str(db[db['name'] == currCard[0]].health.item()),'text':str(db[db['name'] == currCard[0]].text.item())}