# -*- coding: utf-8 -*-

#My imports
import Card_CV
import gameState
import Tracked_Cards

#Linear Algebra
import numpy as np

#Image manipulation
from PIL import ImageGrab
import cv2
import time
import sys


#Internal movement and scheduling for timers
import pyautogui
import imutils


#TODO: Going to need a new approach, create a classifier to find cards instead of raw size values. This will be more obtuse and hopefully allow for more clean tracking.

#Block out unused areas of the screen to avoid noise
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices,400)
    masked = cv2.bitwise_and(img, img, mask = mask)
    return masked

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
    #processed = cv2.Canny(im, threshold1 = 200, threshold2 = 300)
    #processed = cv2.GaussianBlur(processed, (3,3),0)
    #cv2.bitwise_not(im,processed)
    lines = cv2.HoughLinesP(im, 1, np.pi/180, 80, 100,10)
    draw_lines(im, lines)
    return im
    
#Take region of roi containing a card name and pass it to CVBoard.py to describe card
def findNames(orig,gray, game, tracked,tracker, count):
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,3))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    orig = imutils.resize(orig, width = 300)
    if orig.any():
        cv2.bitwise_not(gray)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
        #blur = cv2.GaussianBlur(tophat, (5,5),5)
        gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx = 1, dy = 0, ksize = 1)
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
        #locs = []
        
        #for(i, c) in enumerate(cnts):
            
        #    (x,y,w,h) = cv2.boundingRect(c) #Consider aspect ratio as a means of better identfying cards
        #    if(w > 40 and w < 250) and (h > 18 and h < 35):
        #        locs.append((x - 5,y,w + 10,h + 5))
        #        tracked.append((x- 5, y, w+ 10, h+ 5))
        #    if locs and locs not in tracked:
        #        if count < 10:
        #            tracker.add(cv2.TrackerKCF_create(),orig,locs[0])
        #        game.addToHand(Card_CV.addFromBoard(orig,locs,gray)) #Send roi region (should be the name) to be described
        #        #time.sleep(1)
        #        locs = []

def findCards(orig, gray, tracked, tracker):
    gray = cv2.bilateralFilter(gray, 2, 30,80)
    edged = cv2.Canny(gray, 0,150)
    (_,cnts, _) = cv2.findContours(edged.copy(),  cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    for i,c in enumerate(cnts):
        (x,y,w,h) = cv2.boundingRect(c)
        
        #if len(cnts) <= 10:#$ and len(approx) < 8:
        if(w > 100 and w < 185) and (h > 150 and h < 240):
            cur_roi = (x,y,w,h)
            cv2.rectangle(orig, (x,y), (x+w, y+h), (255,0,0), 4)
            tracked.append((x,y,w,h))
    return tracked
            
    
def main():
    tracked = []
    tracker = cv2.MultiTracker()
    printscreen = ImageGrab.grab(bbox=(0,50, 1024, 800))
    roi_board = cv2.cvtColor(np.array(printscreen), cv2.COLOR_BGR2RGB)
    roi_board_gray = cv2.cvtColor(np.array(roi_board), cv2.COLOR_BGR2GRAY)
    tracked = findCards(roi_board, roi_board_gray, tracked, tracker)
    print(tracked)
    #cv2.SelectROIs("tracker",printscreen,tracked)
    for item in tracked:
        tracker.add(cv2.TrackerKCF_create(),roi_board,item)
    while(True):
        printscreen = ImageGrab.grab(bbox=(0,50, 1024, 800))
        roi_board = cv2.cvtColor(np.array(printscreen), cv2.COLOR_BGR2RGB)
        roi_board_gray = cv2.cvtColor(np.array(roi_board), cv2.COLOR_BGR2GRAY)
        findCards(roi_board, roi_board_gray, tracked, tracker)
        #cv2.SelectROIs("tracker",roi_board,tracked)
        tracker.update(roi_board)
        for obj in range(len(tracker.getObjects())):
            cv2.rectangle(roi_board, tracker.getObjects()[obj], (0,255,0), 4)
        cv2.imshow('window', roi_board)
        if cv2.waitKey(25) & 0xFF == ord('d'):
            game.displayHand()
        if cv2.waitKey(50) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    game = gameState.GameState()
    #_thread.start_new_thread(main())
    main()



### DEPRECATED BUT MIGHT BE USEFUL ###
#vertices = np.array([[100,600],[100,160],[400,160],[400,10],[580,10],[580,160],[1024,160],[1024,450],[750,450],[750,600]])
#roi_board = region_of_interest(np.array(printscreen),[vertices])
#p = process_img(roi_board_gray)
#findCards(roi_board, roi_board_gray, game, tracked,tracker, trackCount)
#ok, boxes = tracker.update(roi_board)
#for newbox in boxes:
#    p1 = (int(newbox[0]), int(newbox[1]))
#    p2 = (int(newbox[0]) + int(newbox[1]), int(newbox[1] + newbox[3]))
#    cv2.rectangle(roi_board, p1,p2,(200,0,0),2)


