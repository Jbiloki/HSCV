# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 11:27:37 2017

@author: Nguyen
"""

#My imports
import CVBoard
import playerClass

#Linear Algebra
import numpy as np

#Image manipulation
from PIL import ImageGrab
import cv2
import time


#Internal movement and scheduling for timers
import pyautogui
import imutils
import sched
import _thread


#Look for lines in the image
#TODO: May not be necessary remove to improve speed
def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0], coords[1]), (coords[2],coords[3]), [0,255,0],1)
    except:
        pass

#Process image
def process_img(im):
    processed = cv2.Canny(im, threshold1 = 200, threshold2 = 300)
    processed = cv2.GaussianBlur(processed, (3,3),10)
    lines = cv2.HoughLinesP(processed, 1, np.pi/180, 180, 50,5)
    draw_lines(processed, lines)
    return processed
    
#Take region of roi containing a card name and pass it to CVBoard.py to describe card
def findCards(orig,gray, player):
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
            (x,y,w,h) = cv2.boundingRect(c) #Consider aspect ratio as a means of better identfying cards
            if(w > 50 and w < 250) and (h > 15 and h < 35):
                cv2.rectangle(p,(x,y), (x+w, y+h),(255,0,0),2)
                locs.append((x - 5,y,w + 10,h + 5))
            if locs:
                player.addToHand(CVBoard.addFromBoard(orig,locs,gray)) #Send roi region (should be the name) to be described
                time.sleep(1)
                locs = []

currHand = []
player = playerHand(currHand)
while(True):
    printscreen = ImageGrab.grab(bbox=(0,100, 1024, 768))
    printscreen = cv2.cvtColor(np.array(printscreen), cv2.COLOR_BGR2RGB)
    vertices = np.array([[100,600],[100,160],[1024,160],[1024,500],[750,500],[750,600]])#,[1100,500],[1100,400]
    roi_board = region_of_interest(printscreen,[vertices])
    roi_board = cv2.cvtColor(np.array(roi_board), cv2.COLOR_BGR2RGB)
    roi_board_gray = cv2.cvtColor(np.array(roi_board), cv2.COLOR_BGR2GRAY)
    p = process_img(roi_board_gray)
    findCards(roi_board, roi_board_gray, player)
    player.displayHand()
    cv2.imshow('window', roi_board)
    cv2.imshow('window2', p)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break