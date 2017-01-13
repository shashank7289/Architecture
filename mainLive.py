'''
Created on Jan 5, 2017

@author: uid38420
'''
from os.path import dirname,realpath
from cv2 import CascadeClassifier
from dlib import shape_predictor
from rotate.rollChk import rollChk
from rotate.rotateLive import rotateLive
from frontalChk.frontalChkLive import frontalChkLive
from livenessDetection.blinkDetection import blinkDetection

if __name__ == '__main__' :
#-------------------------------------------------------------------------------
    #initializations
    #paths
    dirPath = dirname(realpath(__file__))
    faceCascade = CascadeClassifier(dirPath + '/resources/haarcascade_frontalface_default.xml')
    predictor = shape_predictor(dirPath + '/resources/shape_predictor_68_face_landmarks.dat')
#-------------------------------------------------------------------------------
    #affine transforms
    
    #roll angle check
#     rollChk(faceCascade,predictor)
    
    #live rotation
#     rotateLive(faceCascade,predictor)

    #frontal or not
#     frontalChkLive(faceCascade,predictor)

    #liveness detection
#     blinkDetection(faceCascade,predictor)