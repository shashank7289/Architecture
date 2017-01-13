'''
Created on Sep 23, 2016

@author: uid38420
'''
from cv2 import imshow,waitKey,putText,FONT_HERSHEY_SIMPLEX
from frontalChk.frontalCondition import frontalCondition

def frontalChkImg(img):
    
    decision = frontalCondition()
    putText(img, decision, (10, 30),FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
#     imshow('output',img)
#     waitKey(0)
    return decision