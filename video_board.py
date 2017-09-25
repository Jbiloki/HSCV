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

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0], coords[1]), (coords[2],coords[3]), [0,255,0],1)
    except:
        pass

def process_img(im):
    processed = cv2.Canny(im, threshold1 = 200, threshold2 = 300)
    processed = cv2.GaussianBlur(processed, (5,5),0)
    lines = cv2.HoughLinesP(processed, 1, np.pi/180, 180, 50,5)
    draw_lines(processed, lines)
    return processed
    

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
            #ar = w / float(h)
            #if ar > 2 and ar < 10.0:
            if(w > 50 and w < 500) and (h > 15 and h < 35):
                cv2.rectangle(p,(x,y), (x+w, y+h),(255,0,0),2)
                locs.append((x,y,w,h))
            if locs:
                CVBoard.addFromBoard(orig,locs,gray)
        #for(i, (gX,gY,gW,gH)) in enumerate(locs):
        #    group = gray[gY - 5:gY + gH, gX:gX + gW]
        #    cv2.imshow("group", group)

while(True):
    printscreen = ImageGrab.grab(bbox=(0,100, 1024, 768))
    printscreen = cv2.cvtColor(np.array(printscreen), cv2.COLOR_BGR2RGB)
    
    #printscreen_to_numpy = np.array(printscreen.getdata(), dtype='uint8')
    
    #findCards(printscreen)
    #processed = process_img(printscreen_gray)
    vertices = np.array([[100,1000],[100,160],[1200,160],[1200,500],[1100,500],[1100,400]])
    roi_board = region_of_interest(printscreen,[vertices])
    roi_board = cv2.cvtColor(np.array(roi_board), cv2.COLOR_BGR2RGB)
    roi_board_gray = cv2.cvtColor(np.array(roi_board), cv2.COLOR_BGR2GRAY)
    p = process_img(roi_board_gray)
    findCards(roi_board, roi_board_gray)
    cv2.imshow('window', p)
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