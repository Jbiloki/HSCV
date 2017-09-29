# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 17:40:03 2017

@author: Nguyen
"""
import cv2
import numpy as np

class playerHand:
    def __init__(self):
        self.currHand = []
    def displayHand(self):
        print(self.currHand)
    def addToHand(self, card):
        if card is not None:
            self.currHand.append(card)
    def removeFromHand(self, card):
<<<<<<< HEAD
        if card in self.currHand:
            self.currHand.remove(card)
=======
        if card in currHand:
            currHand.remove(card)
>>>>>>> c0d32fac2135702d8308c99569ec4dae4a167085
