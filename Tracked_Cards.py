# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 18:19:55 2017

@author: Nguyen
"""

class Card:
    #TODO: Implement type of card (i.e. spell/weapon/minion)
    def __init__(self, x, y, name = '', attack = '', health = '', text = ''):
        self.x = x
        self.y = y
        self.name = name
        self.attack = attack
        self.health = health
        self.text = text