'''
Created on Jan 5, 2017

@author: uid38420
'''
from os.path import dirname,realpath
from cv2 import CascadeClassifier,imread,cvtColor,COLOR_BGR2YUV,COLOR_YUV2BGR
from dlib import shape_predictor
from helper.findFace import findFace
from preprocessing.preprocessing import histogramEqualization,clahe,gammaCorrection,gicClahe,blurRemove
from rotate.rotateImage import rotateImg
from frontalChk.frontalChkImage import frontalChkImg
from crop.crop import faceCropping

if __name__ == '__main__' :
    
    #initializations
    
    #preprocessing
    gamma = 1.5
    
    #paths
    dirPath = dirname(realpath(__file__))
    faceCascade = CascadeClassifier(dirPath + '/resources/haarcascade_frontalface_default.xml')
    predictor = shape_predictor(dirPath + '/resources/shape_predictor_68_face_landmarks.dat')
    
    img = imread('D:/Codes/TestData/Cropping/Frontalized_image_FEI_database_OpenCV_only_chin_to_chin/11//images/1-05.jpg')
    
    #detect face + landmarks + return face image only
    faceImg = findFace(img,faceCascade,predictor)

    #preprocessing
    
    #histogram equalization
#     histogramEqualization(faceImg)
    
    #Contrast Limited Adaptive Histogram Equalization(CLAHE)
#     clahe(faceImg)
    
    #gamma correction
#     gammaCorrection(faceImg, gamma)
    
    #gamma correction + clahe
#     gicClahe(faceImg,gamma)
    
    #remove blur
#     blurRemove(faceImg)
    
    #affine transforms
    
    #rotate image
#     rotateImg(faceImg)
    
    #frontal or not
#     frontalChkImg(faceImg)
    
    #crop
    faceCropping(faceImg)