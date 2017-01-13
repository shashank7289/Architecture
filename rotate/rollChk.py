'''
Created on Nov 18, 2016

@author: uid38420
'''
from cv2 import VideoCapture,putText,FONT_HERSHEY_SIMPLEX,imshow,waitKey
from numpy import str
from helper.findFace import findFace
from helper.findLandmarks import faceObj
from helper.findRollAngle import findRollAngle

def rollChk(faceCascade,predictor):
    
    cap = VideoCapture(0)
    while True:
        ret,img = cap.read()
    
        #detect face + landmarks + return face image only
        findFace(img,faceCascade,predictor)
        
        #determine roll angle
        angle = findRollAngle(faceObj.leftEyeCorner, faceObj.rightEyeCorner)
        
        putText(img, str(round(angle,2)), (10, 30),FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
        imshow('Roll Angle',img)
        waitKey(1)