'''
Created on Nov 18, 2016

@author: uid38420
'''
from cv2 import VideoCapture,putText,FONT_HERSHEY_SIMPLEX,imshow,waitKey
from numpy import str
from helper.findFace import findFace
from helper.findLandmarks import faceObj
from helper.findRollAngle import findRollAngle
from skimage.transform import rotate

def rotateLive(faceCascade,predictor):
    cap = VideoCapture(0)
    
    while True:
        ret,img = cap.read()
        
        #detect face
        faceImg = findFace(img,faceCascade,predictor)
        
        if faceImg.size !=0:

            #Determine roll angle
            angle = findRollAngle(faceObj.leftEyeCorner, faceObj.rightEyeCorner)
        
            #Rotate image
            rotatedImg = rotate(faceImg, angle)
         
            putText(rotatedImg, str(round(angle,2)), (10, 30),FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            imshow('output',rotatedImg)
            waitKey(1)