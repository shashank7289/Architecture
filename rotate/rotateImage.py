'''
Created on Nov 18, 2016

@author: uid38420
'''
from cv2 import imshow,waitKey
from helper.findLandmarks import faceObj
from helper.findRollAngle import findRollAngle
from skimage.transform import rotate

def rotateImg(faceImg):
        
    #determine roll angle
    angle = findRollAngle(faceObj.leftEyeCorner, faceObj.rightEyeCorner)
    
    #rotate image
    rotatedImg = rotate(faceImg, angle)
        
#     imshow('output',rotatedImg)
#     waitKey(0)
    return rotatedImg