'''
Created on Sep 23, 2016

@author: uid38420
'''
from cv2 import VideoCapture,imshow,waitKey,putText,FONT_HERSHEY_SIMPLEX
from helper.findFace import findFace
from helper.findLandmarks import findLandmarks
from frontalCondition import frontalCondition

def frontalChkLive(faceCascade,predictor):

    cap = VideoCapture(0)
    while True:
        ret,img = cap.read()
        
        #detect face
        faces = findFace(img,faceCascade,predictor)
                  
        decision = frontalCondition()
        putText(img, decision, (10, 30),FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
        imshow('output',img)
        waitKey(1)