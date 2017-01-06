'''
Created on Jan 5, 2017

@author: uid38420
'''
from numpy import arctan2,pi

def findRollAngle(p1, p2):
    if (p1[1]> p2[1]):
        p1[1] = p1[1]-p2[1]
        p2[1] = p2[1]-p2[1]
    elif (p2[1]> p1[1]):
        p2[1] = p2[1]-p1[1]
        p1[1] = p1[1]-p1[1]
    xDiff = p2[0] - p1[0]
    yDiff = p2[1] - p1[1]
    angle = arctan2(yDiff,xDiff) * (180 / pi)
    if angle > 180:
        angle = 360-angle
        angle = -angle
    return angle