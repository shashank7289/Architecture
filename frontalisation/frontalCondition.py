'''
Created on Jan 5, 2017

@author: uid38420
'''
from helper.findLandmarks import faceObj
from numpy import argmin,argmax,split,subtract,sqrt,square

def frontalCondition():
    ratio = score = awayScore = 0.0
    
    #Condition 1 distance between outer eye point and edge face outline
    #difference of face outline and outermost point of left eye
    minPos = argmin(faceObj.leftEye,axis=0)
    minX = minPos[0]
    minX = split(faceObj.leftEye[minX],[1])
    outline = split(faceObj.outline[0],[1])
    diff1 = subtract(outline,minX)
    diff1 = sqrt(square(diff1[0]) + square(diff1[1]))
       
    #difference of face outline and outermost point of right eye
    maxPos = argmax(faceObj.rightEye,axis=0)
    max = maxPos[0]
    max = split(faceObj.rightEye[max],[1])
    outline = split(faceObj.outline[16],[1])
    diff2 = subtract(outline,max)
    diff2 = sqrt(square(diff2[0]) + square(diff2[1]))
       
    if (diff1 >= diff2):
        ratio = diff2/diff1
    else:
        ratio = diff1/diff2
    if (ratio >= 0.65):
        score = score  + 0.3
    elif(ratio <= 0.3):
        awayScore = awayScore + 0.3

    #Condition 2 distance between bottom nose point and corresponding point on the chin
    min = 0; max = 1000
    point = [0,0]
     
    noseCentre = split(faceObj.nose[3],[1])
    outlineX = split(faceObj.outline[:,0],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
    outlineY = split(faceObj.outline[:,-1],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
     
    for i in xrange(17):
        #top point
        if((outlineY[i] <= (noseCentre[1]+2)) and (outlineY[i] >= min)):            
            min = outlineY[i]
            point[0] = i
        #bottom point
        elif((outlineY[i] >= (noseCentre[1]-2)) and (outlineY[i] <= max)):
            max = outlineY[i]
            point[1] = i
    
    diffX = noseCentre[0] - outlineX[point[0]]
    diffY = noseCentre[1] - outlineY[point[0]]
    diff1 = sqrt(square(diffX) + square(diffY))
     
    diffX = noseCentre[0] - outlineX[point[1]]
    diffY = noseCentre[1] - outlineY[point[1]]
    diff2 = sqrt(square(diffX) + square(diffY))
     
    if (diff1 >= diff2):
        ratio = diff2/diff1
    else:
        ratio = diff1/diff2
    if (ratio >= 0.65):
        score = score  + 0.3
    elif(ratio <= 0.3):
        awayScore = awayScore + 0.3
    
    #Condition 3 distance between outer eyebrow point and extreme end chin points
    diff1 = subtract(faceObj.outline[0],faceObj.eyeBrows[0])
    diff1 = sqrt(square(diff1[0]) + square(diff1[1]))
     
    diff2 = subtract(faceObj.outline[16],faceObj.eyeBrows[9])
    diff2 = sqrt(square(diff2[0]) + square(diff2[1]))
     
    if (diff1 >= diff2):
        ratio = diff2/diff1
    else:
        ratio = diff1/diff2
    if (ratio >= 0.65):
        score = score  + 0.3
    elif(ratio <= 0.3):
        awayScore = awayScore + 0.3
    
    #decision making
    decision = ""
    if (score> 0.5):
        decision = "frontal"
    else:
        decision = "non-frontal"
    return decision,awayScore