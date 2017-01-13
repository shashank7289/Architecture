'''
Created on Jan 5, 2017

@author: uid38420
'''
from os.path import dirname,realpath
from cv2 import CascadeClassifier
from dlib import shape_predictor
from numpy import asarray
from scipy.io import loadmat
from rotate.rollChk import rollChk
from rotate.rotateLive import rotateLive
from frontalisation.frontalChkLive import frontalChkLive
from frontalisation import frontalize
from frontalisation.frontalisationLive import frontalisationLive
from livenessDetection.blinkDetection import blinkDetection

if __name__ == '__main__' :
#-------------------------------------------------------------------------------
    #initializations
    #paths
    dirPath = dirname(realpath(__file__))
    faceCascade = CascadeClassifier(dirPath + '/resources/haarcascade_frontalface_default.xml')
    predictor = shape_predictor(dirPath + '/resources/shape_predictor_68_face_landmarks.dat')
    model3D = frontalize.ThreeD_Model(dirPath + "/resources/model3Ddlib.mat", 'model_dlib')
    eyemask = asarray(loadmat('resources/eyemask.mat')['eyemask']) #mask to exclude eyes from symmetry
#-------------------------------------------------------------------------------
    #affine transforms
    
    #roll angle check
#     rollChk(faceCascade,predictor)
    
    #live rotation
    rotateLive(faceCascade,predictor)

    #frontal or not
#     frontalChkLive(faceCascade,predictor)

    #frontalisationImage
#     frontalisationLive(faceCascade,predictor,model3D,eyemask)

    #liveness detection
#     blinkDetection(faceCascade,predictor)