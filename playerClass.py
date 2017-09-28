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
        if card in self.currHand:
            self.currHand.remove(card)