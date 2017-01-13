'''
Created on Nov 25, 2016

@author: uid38420
'''
from dlib import rectangle
from numpy import insert,zeros,array
from cv2 import circle

class face(object):
    def __init__(self):
        self.lMarksFloat = self.lMarksInt = []
        self.outline = self.eyeBrows = self.nose = self.leftEye = self.rightEye = self.mouth = []
        self.leftEyeCorner = self.rightEyeCorner = 0
        
faceObj = face();
        
#find Landmarks
def findLandmarks(img,face,predictor):
    
    lMarks = zeros((68, 2))
#     lMarks = []
    for (x,y,w,h) in face:
        d = rectangle(x,y,x+w,y+h)
        i = 0
        for p in predictor(img, d).parts():
            lMarks = insert(lMarks, i, [p.x, p.y], 0)
            i = i+1
#             circle(img,(p.x,p.y),1,(0,255,0), 1)
    faceObj.lMarksInt = lMarks[:68].astype(int)
    faceObj.lMarksFloat = lMarks[:68].astype(float)
    faceParts(faceObj.lMarksInt,faceObj.lMarksFloat)

def faceParts(lMarksInt,lMarksFloat):
    faceObj.outline = array(lMarksFloat[:17])
    faceObj.eyeBrows = array(lMarksFloat[17:27])
    faceObj.nose = array(lMarksFloat[27:36])
    faceObj.leftEye = array(lMarksFloat[36:42])
    faceObj.rightEye = array(lMarksFloat[42:48])
    faceObj.leftEyeCorner = array(lMarksInt[36])
    faceObj.rightEyeCorner = array(lMarksInt[45])
    faceObj.mouth = array(lMarksFloat[48:])