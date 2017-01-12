'''
Created on Nov 25, 2016

@author: uid38420
'''
from dlib import rectangle
from numpy import insert,zeros,array
from cv2 import circle

class face(object):
    def __init__(self):
        self.outline = self.eyeBrows = self.nose = self.leftEye = self.rightEye = self.mouth = []
        self.leftEyeCorner = self.rightEyeCorner = 0
        
faceObj = face();
        
#find Landmarks
def findLandmarks(img,faces,predictor):
    
    lMarks = zeros((68, 2))
    
    for (x,y,w,h) in faces:
        d = rectangle(x,y,x+w,y+h)
        i = 0
        for p in predictor(img, d).parts():
            lMarks = insert(lMarks, i, [p.x, p.y], 0)
            i = i+1
#             circle(img,(p.x,p.y),1,(0,255,0), 1)
    lMarks = lMarks.astype(int)
    faceParts(lMarks)

def faceParts(lMarks):
    faceObj.outline = array(lMarks[:17])
    faceObj.eyeBrows = array(lMarks[17:27])
    faceObj.nose = array(lMarks[27:36])
    faceObj.leftEye = array(lMarks[36:42])
    faceObj.rightEye = array(lMarks[42:48])
    faceObj.leftEyeCorner = array(lMarks[36])
    faceObj.rightEyeCorner = array(lMarks[45])
    faceObj.mouth = array(lMarks[48:])