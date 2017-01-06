import frontalize
import facial_feature_detector as feature_detection
import camera_calibration as calib
import scipy.io as io
import cv2
import numpy as np
import os
import sys
import time
import ctypes
import copy
import math
this_path = os.path.dirname(os.path.abspath(__file__))

VIDEO = 1

def demo():
    model3D = frontalize.ThreeD_Model(this_path + "/model_file/model3Ddlib.mat", 'model_dlib')

    if(VIDEO):
        cap = cv2.VideoCapture(0)

        if cap.isOpened():
            ret, img = cap.read()
        else:
            ret = False

        while ret:
            ret, img = cap.read()
            # start_col = 0.2*640;
            # end_col = 640 - start_col;
            # img = img[start_col:end_col,0:480]
            cv2.imshow("Live Video---press Space bar to capture---Esc to exit", img)
            key = cv2.waitKey(1)
            if key == 32:  # Press Space key for capturing
                break
            elif key == 27:
                sys.exit()
    else:
        img = cv2.imread("D:\\Projects\\Database\\HeadPoseImageDatabase\\Person04\\person04131-15-30.jpg", 1)
        #D:\\Projects\\Database\\special case\\image_1.jpg
        #D:\\Projects\\Image Processing\\Visual Studio Projects\\Dlib_Video\\Dlib_Video\\experiment\\104.jpg
        #D:\\Projects\\Database\\HeadPoseImageDatabase\\Person01\\person01146+0+0.jpg
        #D:\\Projects\\Database\\Subject01_10\\Subject01\\A_01_0.Jpg
        # start_col = 0.2*640;
        # end_col = 640 - start_col;
        # img = img[start_col:end_col,1:480]

    cv2.imshow("Original Image",img)
    clone_image = copy.copy(img)

    # Landmarks & Face detection
    start_time = time.time()
    no_of_faces, lmarks, blank_image = feature_detection.get_landmarks(clone_image,0)# 0-before frontalisation
    execution_time = (time.time() - start_time) * 1000
    print "Face detection time in milliseconds : ", execution_time

    if no_of_faces:
        for x in xrange(lmarks.shape[1]):
            cv2.circle(clone_image, (lmarks[0][x, 0], lmarks[0][x, 1]), 1, (0, 255, 0), 1, 0, 0)

        # Check if face is frontal or not
        x,y,frontal_status,non_frontal_limit = frontal_pose_check(no_of_faces, lmarks)
        if non_frontal_limit < 0.6:
            cv2.imshow("Landmark Image", clone_image)

            if (frontal_status == 0):

                # FRONTALISATION

                # 1. CAMERA CALIBRATION
                proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, lmarks[0])

                # 2. LOAD MASK TO EXCLUDE EYES FROM SYMMETRY
                eyemask = np.asarray(io.loadmat('model_file/eyemask.mat')['eyemask'])

                start_time = time.time()
                # 3. PERFORM FRONTALISATION
                frontal_raw, frontal_sym = frontalize.frontalize(blank_image, proj_matrix, model3D.ref_U, eyemask)
                execution_time = (time.time() - start_time) * 1000

                print "Frontalization execution time in milliseconds : ", execution_time
                cv2.imshow("Frontalized with symmetry", frontal_sym))
                # ***************************************************************

                # Landmarks on frontalised face
                no_of_faces_frontal , lmarks = feature_detection.get_landmarks(frontal_sym,1) #1-after frontalisation
                if no_of_faces_frontal:
                    for x in xrange(lmarks.shape[1]):
                        cv2.circle(frontal_sym, (lmarks[0][x, 0], lmarks[0][x, 1]), 1, (0, 255, 0), 2, 0, 0)

                    cv2.imshow("Landmark on Frontalised Image", frontal_sym)
                    cv2.waitKey(0)
                    cv2.close("all")
                    cv2.destroyAllWindows()
                else:
                    ctypes.windll.user32.MessageBoxA(0, "No face in frontal image", "Message", 1)
                    print "No face in frontal image"
            else:
                print "Its already a frontal face"
                ctypes.windll.user32.MessageBoxA(0, "Its already a frontal face", "Message", 1)
                # cv2.imwrite("D:\\Projects\\Image Processing\\Python projects\\Face_frontalisation_python\\frontal_face.png",frontal_sym)
                cv2.waitKey(0)
                cv2.close("all")
                cv2.destroyAllWindows()
        else:
            # cv2.putText(clone_image, "Please face the camera", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),2)
            cv2.imshow("Landmark Image", clone_image)
            ctypes.windll.user32.MessageBoxA(0, "Please face the camera", "Message", 1)
            cv2.waitKey(0)
            cv2.close("all")
            cv2.destroyAllWindows()
    else:
        print "No face detected"
        ctypes.windll.user32.MessageBoxA(0, "No face detected", "Message", 1)
        cv2.waitKey(0)
        cv2.close("all")
        cv2.destroyAllWindows()



def frontal_pose_check(no_of_faces, lmarks):
    for i in range(no_of_faces):
        left_eye = np.array(([lmarks[0][36][0], lmarks[0][36][1]],
                             [lmarks[0][37][0], lmarks[0][37][1]],
                             [lmarks[0][38][0], lmarks[0][38][1]],
                             [lmarks[0][39][0], lmarks[0][39][1]],
                             [lmarks[0][40][0], lmarks[0][40][1]],
                             [lmarks[0][41][0], lmarks[0][41][1]]))

        right_eye = np.array(([lmarks[0][42][0], lmarks[0][42][1]],
                              [lmarks[0][43][0], lmarks[0][43][1]],
                              [lmarks[0][44][0], lmarks[0][44][1]],
                              [lmarks[0][45][0], lmarks[0][45][1]],
                              [lmarks[0][46][0], lmarks[0][46][1]],
                              [lmarks[0][47][0], lmarks[0][47][1]]))

        print "Condition 1 distance between outer eye point and edge face outline"
        minimum_pos = np.argmin(left_eye[:, 0])
        maximum_pos = np.argmax(right_eye[:, 0])
        diff_x = np.abs(lmarks[0][0][0] - left_eye[minimum_pos, 0])
        diff_y = np.abs(lmarks[0][0][1] - left_eye[minimum_pos, 1])
        diff1 = np.sqrt((diff_x * diff_x) + (diff_y * diff_y))
        left_eye_position = minimum_pos
        right_eye_position = maximum_pos

        diff_x = np.abs(lmarks[0][16][0] - right_eye[maximum_pos, 0])
        diff_y = np.abs(lmarks[0][16][1] - right_eye[maximum_pos, 1])
        diff2 = np.sqrt((diff_x * diff_x) + (diff_y * diff_y))

        if (diff1 >= diff2):
            ratio = diff2 / diff1
        else:
            ratio = diff1 / diff2

        print "ratio :  ", ratio
        score = 0.0
        away_score = 0.0
        if (ratio >= 0.65):
            score = score + 0.3
        elif(ratio <= 0.3):
            away_score = away_score + 0.3

        print "Condition 2 distance between bottom nose point and corresponding point on the chin"
        value = lmarks[0][30][1]  # Nose point
        minimum = 0
        maximum = 1000
        for i in range(17):

            # Top point
            if ((lmarks[0][i][1] <= (value + 2)) & (lmarks[0][i][1] >= minimum)):
                minimum = lmarks[0][i][1]
                point_0 = i

            # Bottom point
            elif ((lmarks[0][i][1] >= (value - 2)) & (lmarks[0][i][1] <= maximum)):
                maximum = lmarks[0][i][1]
                point_1 = i

        diff_x = np.abs(lmarks[0][30][0] - lmarks[0][point_0][0])
        diff_y = np.abs(lmarks[0][30][1] - lmarks[0][point_0][1])
        diff1 = np.sqrt((diff_x * diff_x) + (diff_y * diff_y))

        diff_x = np.abs(lmarks[0][30][0] - lmarks[0][point_1][0])
        diff_y = np.abs(lmarks[0][30][1] - lmarks[0][point_1][1])
        diff2 = np.sqrt((diff_x * diff_x) + (diff_y * diff_y))

        if (diff1 >= diff2):
            ratio = diff2 / diff1
        else:
            ratio = diff1 / diff2

        print "ratio :  ", ratio
        if (ratio >= 0.65):
            score = score + 0.3
        elif (ratio <= 0.3):
            away_score = away_score + 0.3

        print "Condition 3 distance between outer eyebrow point and extreme end chin points"
        diff_x = np.abs(lmarks[0][0][0] - lmarks[0][17][0])
        diff_y = np.abs(lmarks[0][0][1] - lmarks[0][17][1])
        diff1 = np.sqrt((diff_x * diff_x) + (diff_y * diff_y))

        diff_x = np.abs(lmarks[0][16][0] - lmarks[0][26][0])
        diff_y = np.abs(lmarks[0][16][1] - lmarks[0][26][1])
        diff2 = np.sqrt((diff_x * diff_x) + (diff_y * diff_y))

        if (diff1 >= diff2):
            ratio = diff2 / diff1
        else:
            ratio = diff1 / diff2

        print "ratio :  ", ratio
        if (ratio >= 0.65):
            score = score + 0.3
        elif (ratio <= 0.3):
            away_score = away_score + 0.3

    print "score : ", score

    if (score > 0.5):
        return int((lmarks[0][21][0] + lmarks[0][22][0]) / 2), lmarks[0][27][1], 1,away_score
    else:
        return int((lmarks[0][21][0] + lmarks[0][22][0]) / 2), lmarks[0][27][1], 0,away_score


if __name__ == "__main__":
    demo()
