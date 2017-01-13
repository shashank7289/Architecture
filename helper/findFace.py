'''
Created on Jan 5, 2017

@author: uid38420
'''
from numpy import size,array
from helper.findLandmarks import findLandmarks
from helper.faceOnly import faceOnly
from cv2 import cvtColor,COLOR_BGR2GRAY

def findFace(img,faceCascade,predictor):
    faces = faceCascade.detectMultiScale(img, 1.3, 5)
    noOfFaces = size(faces,0)
     
    #discard smaller faces
    if noOfFaces >= 1:
        for (x,y,w,h) in faces:
            if h < 100:
                pass
            elif h > 100:
                #face only
                faceImg = faceOnly(img,faces)
                gray = cvtColor(faceImg, COLOR_BGR2GRAY)

                #determine face landmark points
                faceLmarks = faceCascade.detectMultiScale(gray, 1.3, 5)
                findLandmarks(gray,faceLmarks,predictor)
                return faceImg
    else:
        return array([])
#         return img