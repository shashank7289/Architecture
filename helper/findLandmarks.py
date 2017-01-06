'''
Created on Nov 25, 2016

@author: uid38420
'''
import dlib
import os 
import numpy as np
# import cv2

class face(object):
    def __init__(self):
        self.outline = self.eyeBrows = self.nose = self.leftEye = self.rightEye = self.mouth = []
        self.leftEyeCorner = self.rightEyeCorner = 0
        
faceObj = face();
        
#find Landmarks
def findLandmarks(img,faces,predictor):
    
    lMarks = np.zeros((68, 2))
    
    for (x,y,w,h) in faces:
        d = dlib.rectangle(x,y,x+w,y+h)
        i = 0
        for p in predictor(img, d).parts():
            lMarks = np.insert(lMarks, i, [p.x, p.y], 0)
            i = i+1
#             cv2.circle(img,(p.x,p.y),1,(0,255,0), 2)
    lMarks = lMarks.astype(int)
    faceParts(lMarks)

def faceParts(lMarks):
    faceObj.outline = np.array(lMarks[:17])
    faceObj.eyeBrows = np.array(lMarks[17:27])
    faceObj.nose = np.array(lMarks[27:36])
    faceObj.leftEye = np.array(lMarks[36:42])
    faceObj.rightEye = np.array(lMarks[42:48])
    faceObj.leftEyeCorner = np.array(lMarks[36])
    faceObj.rightEyeCorner = np.array(lMarks[45])
    faceObj.mouth = np.array(lMarks[48:])