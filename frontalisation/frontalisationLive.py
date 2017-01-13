'''
Created on Jan 13, 2017

@author: uid38420
'''
from sys import exit
from cv2 import VideoCapture,imshow,waitKey,putText,FONT_HERSHEY_SIMPLEX
from helper.findFace import findFace
from helper.findLandmarks import faceObj
from frontalize import frontalize
from camera_calibration  import estimate_camera
from frontalisation.frontalCondition import frontalCondition

def frontalisationLive(faceCascade,predictor,model3D,eyemask):
    cap = VideoCapture(0)
    while True:
        ret, img = cap.read()
        imshow("Live Video: press Space bar to capture or Esc to exit", img)
        key = waitKey(1)
        if key == 32:  # Press Space key for capturing
            break
        elif key == 27:
            exit()

    #detect face
    faceImg = findFace(img,faceCascade,predictor)

    #frontal or not
    frontalDecision,awayScore = frontalCondition()

    # Check if face is frontal or not
    if awayScore < 0.6:
        if frontalDecision == "non-frontal":
            #camera calibration
            proj_matrix, camera_matrix, rmat, tvec = estimate_camera(model3D, faceObj.lMarksFloat)

            #frontalisationImage
            frontal_raw, frontal_sym = frontalize(faceImg, proj_matrix, model3D.ref_U, eyemask)
            imshow("Frontalized with symmetry", frontal_sym)
            waitKey(0)
        else:
            putText(faceImg, "Already a frontal face", (10, 40), FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)
            imshow("", faceImg)
            waitKey(0)
    else:
        putText(faceImg, "Please face the camera", (10, 40), FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)
        imshow("", faceImg)
        waitKey(0)