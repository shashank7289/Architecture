'''
Created on Jan 5, 2017

@author: uid38420
'''
from os import path
from cv2 import CascadeClassifier,imread
from dlib import shape_predictor
from helper.findFace import faceNotDetected
from helper.findFace import findFace
from helper.findLandmarks import findLandmarks
from helper.faceOnly import faceOnly
from rotate.rotateImage import rotateImg
from frontalChk.frontalChkImage import frontalChkImg
from crop.crop import faceCropping

if __name__ == '__main__' :
    
    dirPath = path.dirname(path.realpath(__file__))
    predictorPath = dirPath + '/resources/shape_predictor_68_face_landmarks.dat'
    predictor = shape_predictor(predictorPath)
    faceCascade = CascadeClassifier(dirPath + '/resources/haarcascade_frontalface_default.xml')
    
    img = imread('D:/Codes/TestData/Cropping/Frontalized_image_FEI_database_OpenCV_only_chin_to_chin/11//images/1-05.jpg')
    
    try:
        #detect face
        faces = findFace(img,faceCascade)
        
        #determine face landmark points
        findLandmarks(img,faces,predictor)
         
        #face on white background
        faceImg = faceOnly(img,faces)
        
        #rotate image
#         rotate(faceImg)
        
        #frontal or not
#         frontalChkImg(img)
        
        #crop
        faceCropping(img)
        
        
    except faceNotDetected:
        print "No face detected"