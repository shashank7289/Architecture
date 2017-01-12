'''
Created on Nov 18, 2016

@author: uid38420
'''
import cv2
from numpy import str
from helper.findLandmarks import faceObj
from helper.findRollAngle import findRollAngle
from skimage.transform import rotate
    
def faceOnly(img,faces):
    faceImg = 0
    for (x,y,w,h) in faces:
        multiplier = 0.35
        faceImg = img[y-multiplier*h:y+h+multiplier*h,x-multiplier*h:x+w+multiplier*h]
    return faceImg

def rotateImg(faceImg):
        
    #determine roll angle
    angle = findRollAngle(faceObj.leftEyeCorner, faceObj.rightEyeCorner)
    
    #rotate image
    rotatedImg = rotate(faceImg, angle)
        
    cv2.imshow('output',rotatedImg)
    cv2.waitKey(0)
    return rotatedImg