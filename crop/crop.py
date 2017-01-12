'''
Created on Oct 5, 2016

@author: uid38420
'''
from math import sin,cos,pi
from numpy import copy,int,array
from helper.findLandmarks import faceObj
from cv2 import estimateRigidTransform,warpAffine,imshow,waitKey

''' Compute similarity transform given two sets of two points.
    OpenCV requires 3 pairs of corresponding points.
    We are faking the third one.'''
def similarityTransform(inPoints, outPoints):
    s60 = sin(60*pi/180);
    c60 = cos(60*pi/180);  
  
    inPts = copy(inPoints).tolist();
    outPts = copy(outPoints).tolist();
    
    xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0];
    yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1];
    
    inPts.append([int(xin), int(yin)]);
    
    xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0];
    yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1];
    
    outPts.append([int(xout), int(yout)]);
    
    tform = estimateRigidTransform(array([inPts]), array([outPts]), False);
    return tform;

def faceCropping(img):
    
    # Dimensions of output image
    w = 128;
    h = 128;
    
    # Eye corners
    eyeFactor = 0.2 #to place eyes at particular places
    eyecornerDst = [(int(eyeFactor*w ), int(h/3)), (int((1-eyeFactor)*w), int(h/3))];

    # Corners of the eye in input image
    eyecornerSrc  = [faceObj.leftEyeCorner, faceObj.rightEyeCorner]
    
    # Compute similarity transform
    tform = similarityTransform(eyecornerSrc, eyecornerDst)
    
    # Apply similarity transformation
    imgWarped = warpAffine(img, tform, (w,h));
    imgMorphed = imgWarped * 255
    
    imshow("Cropped image",imgWarped)
    waitKey(0)
    
    return imgMorphed