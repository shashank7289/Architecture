'''
Created on Nov 18, 2016

@author: uid38420
'''
import cv2
from numpy import str
from helper.findFace import findFace
from helper.findLandmarks import faceObj
from helper.findLandmarks import findLandmarks
from helper.faceOnly import faceOnly
from helper.findRollAngle import findRollAngle
from skimage.transform import rotate

def rotateLive(faceCascade,predictor):
    cap = cv2.VideoCapture(0)
    
    while True:
        ret,img = cap.read()
        
        #detect face
        faces = findFace(img,faceCascade)

        #determine face landmark points
        findLandmarks(img,faces,predictor)

        #Determine roll angle
        angle = findRollAngle(faceObj.leftEyeCorner, faceObj.rightEyeCorner)
        
        #face on white background
        faceImg = faceOnly(img,faces)
        
        #Rotate image
        rotatedImg = rotate(faceImg, angle)
        
        cv2.putText(img, str(round(angle,2)), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
#         cv2.imshow('original',img)
        cv2.imshow('output',rotatedImg)
        cv2.waitKey(1)