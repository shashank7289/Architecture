'''
Created on Jan 13, 2017

@author: uid38420
'''
from helper.findLandmarks import faceObj
from frontalize import frontalize
from camera_calibration  import estimate_camera
from cv2 import imshow,waitKey,putText,FONT_HERSHEY_SIMPLEX
from frontalisation.frontalCondition import frontalCondition

def frontalisationImage(img,model3D,eyemask):

    #frontal or not
    frontalDecision,awayScore = frontalCondition()

    # Check if face is frontal or not
    if awayScore < 0.6:
        if frontalDecision == "non-frontal":
            #camera calibration
            proj_matrix, camera_matrix, rmat, tvec = estimate_camera(model3D, faceObj.lMarksFloat)

            #frontalisationImage
            frontal_raw, frontal_sym = frontalize(img, proj_matrix, model3D.ref_U, eyemask)
            imshow("Frontalized with symmetry", frontal_sym)
            waitKey(0)
        else:
            putText(img, "Already a frontal face", (10, 40), FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)
            imshow("", img)
            waitKey(0)
    else:
        putText(img, "Please face the camera", (10, 40), FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)
        imshow("", img)
        waitKey(0)