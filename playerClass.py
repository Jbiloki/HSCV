# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 17:40:03 2017

@author: Nguyen
"""

class playerHand:
    def __init__(self, currHand):
        self.currHand = currHand
    def displayHand(self):
        print(self.currHand)
    def addToHand(self, card):
        if card is not None:
            self.currHand.append(card)
    def removeFromHand(self, card):
        if card in currHand:
            currHand.remove(card)
#Block out unused areas of the screen to avoid noise
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked