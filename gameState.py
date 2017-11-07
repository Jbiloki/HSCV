# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 17:40:03 2017

@author: Nguyen
"""
import cv2
import numpy as np

class GameState:
    def __init__(self):
        self.playerHand = []
        self.rivalBoard = []
        self.playerBoard = []    
        self.tracked_cards = []
    
    #Player Methods
    def displayHand(self):
        print(self.playerHand)
    def addToHand(self, card):
        if card is not None:
            self.playerHand.append(card)
    def removeFromHand(self, card):
        if card in self.playerHand:
            self.playerHand.remove(card)
    
    #Game Board Methods
    def displayBoard(self):
        print("Rival: ",self.rivalBoard, "\n Your Board: ", self.playerBoard)
    def addToRival(self, card):
        if card is not None:
            self.rivalBoard.append(card)
    def addToPlayer(self, card):
        if card is not None:
            self.playerBoard.append(card)
