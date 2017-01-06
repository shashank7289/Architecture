'''
Created on Jan 5, 2017

@author: uid38420
'''
from numpy import size
from cv2 import imshow,waitKey,putText,FONT_HERSHEY_SIMPLEX

class faceNotDetected(Exception):
    pass

def findFace(img,faceCascade):
    faces = faceCascade.detectMultiScale(img, 1.3, 5)
    noOfFaces = size(faces, 0)
    if (noOfFaces == 1):
        return faces
    
    #discard smaller faces
    elif noOfFaces > 1:
        for (x,y,w,h) in faces:
            if h < 100:
                pass
            elif h > 100:
                return faces
    else:
        putText(img, "No face detected", (10, 40), FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        imshow("Output Image", img)
        waitKey(1)
#         raise faceNotDetected