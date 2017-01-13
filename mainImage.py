'''
Created on Jan 5, 2017

@author: uid38420
'''
from os.path import dirname,realpath
from cv2 import CascadeClassifier,imread
from dlib import shape_predictor
from numpy import asarray
from scipy.io import loadmat
from helper.findFace import findFace
from preprocessing.preprocessing import histogramEqualization,clahe,gammaCorrection,gicClahe,blurRemove
from rotate.rotateImage import rotateImg
from frontalisation.frontalChkImage import frontalChkImg
from crop.crop import faceCropping
from frontalisation import frontalize
from frontalisation.frontalisationImage import frontalisationImage

if __name__ == '__main__' :
#-------------------------------------------------------------------------------
    #initializations
    
    #preprocessing
    gamma = 1.5
    
    #paths
    dirPath = dirname(realpath(__file__))
    faceCascade = CascadeClassifier(dirPath + '/resources/haarcascade_frontalface_default.xml')
    predictor = shape_predictor(dirPath + '/resources/shape_predictor_68_face_landmarks.dat')
    model3D = frontalize.ThreeD_Model(dirPath + "/resources/model3Ddlib.mat", 'model_dlib')
    eyemask = asarray(loadmat('resources/eyemask.mat')['eyemask']) #mask to exclude eyes from symmetry
    
    img = imread('D:/Codes/TestData/Cropping/Frontalized_image_FEI_database_OpenCV_only_chin_to_chin/11//images/1-05.jpg')
    
    #detect face + landmarks + return face image only
    faceImg = findFace(img,faceCascade,predictor)
#-------------------------------------------------------------------------------
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
#-------------------------------------------------------------------------------
    #affine transforms
    
    #rotate image
#     rotateImg(faceImg)
    
    #crop
#     faceCropping(faceImg)

    #frontal or not
#     frontalChkImg(faceImg)

    #frontalisationImage
    frontalisationImage(faceImg,model3D,eyemask)