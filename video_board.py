# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 11:27:37 2017

@author: Nguyen
"""

import numpy as np
from PIL import ImageGrab
import CVBoard
import cv2
import time
import pyautogui
import imutils

def findCards(orig,gray):
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,3))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    orig = imutils.resize(orig, width = 300)
    if orig.any():
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
            #CVBoard.addFromBoard(orig,locs,gray)
        for(i, (gX,gY,gW,gH)) in enumerate(locs):
            group = gray[gY - 5:gY + gH, gX:gX + gW]
            cv2.imshow("group", group)

while(True):
    printscreen = ImageGrab.grab(bbox=(0,40, 1024, 768))
    printscreen = cv2.cvtColor(np.array(printscreen), cv2.COLOR_BGR2RGB)
    printscreen_gray = cv2.cvtColor(np.array(printscreen), cv2.COLOR_BGR2GRAY)
    #printscreen_to_numpy = np.array(printscreen.getdata(), dtype='uint8')
    
    #findCards(printscreen)
    #processed = process_img(printscreen_gray)
    findCards(printscreen, printscreen_gray)
    cv2.imshow('window', printscreen)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    
    
    
    
    
'''OLDPROCESSING    
def process_img(orig_img):
    #processed = cv2.Canny(orig_img,threshold1 = 400, threshold2 = 500)
    #blurred = cv2.GaussianBlur(processed, (9,9), 10)
    #thresh = cv2.threshold(blurred, 60, 355, cv2.THRESH_BINARY)[1]
    #cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    #for(i,c) in enumerate(cnts):
    #    (x,y,w,h) = cv2.boundingRect(c)
    #    if(w > 60 and w < 500) and (h > 20 and h < 30):#if(w > 25 and w < 800) and (h > 25 and h < 800):
    #        cv2.rectangle(printscreen, (x,y), (x+w, y+h), (255,0,0), 2)
    findCards(orig_img)
    #cv2.imshow("passed", printscreen[y:y+h, x:x+w])
    #CVBoard.addFromBoard(orig_img)
    return processed
'''