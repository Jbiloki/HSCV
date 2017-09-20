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

while(True):
    printscreen = ImageGrab.grab(bbox=(0,40, 1024, 768))
    #printscreen_to_numpy = np.array(printscreen.getdata(), dtype='uint8')
    cv2.imshow('window', cv2.cvtColor(np.array(printscreen),cv2.COLOR_BGR2RGB))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break