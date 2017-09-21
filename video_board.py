# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 11:27:37 2017

@author: Nguyen
"""

import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui

def process_img(orig_img):
    processed = cv2.Canny(orig_img,threshold1 = 400, threshold2 = 500)
    return processed

while(True):
    printscreen = ImageGrab.grab(bbox=(0,40, 1024, 768))
    printscreen = cv2.cvtColor(np.array(printscreen), cv2.COLOR_BGR2RGB)
    #printscreen_to_numpy = np.array(printscreen.getdata(), dtype='uint8')
    processed = process_img(printscreen)
    cv2.imshow('window', processed)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break