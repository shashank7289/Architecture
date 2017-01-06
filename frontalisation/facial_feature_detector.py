__author__ = 'Douglas'

import dlib
import os
import numpy as np
import cv2

this_path = os.path.dirname(__file__)
OPENCV_FACE_DETECTION = 0

def _shape_to_np(shape):
    xy = []
    for i in range(68):
        xy.append((shape.part(i).x, shape.part(i).y,))
    xy = np.asarray(xy, dtype='float32')
    return xy


def get_landmarks(img,flag):

    if(OPENCV_FACE_DETECTION):
        # Load face detection cascade
         faceCascade = cv2.CascadeClassifier("D:/Projects/Image Processing/Python projects/Dlib_frontalisation_python/model_file/haarcascade_frontalface_alt.xml")
         faces = faceCascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
        
    else:
        detector = dlib.get_frontal_face_detector()

        predictor = dlib.shape_predictor(
            "D:/Projects/Image Processing/Python projects/Dlib_frontalisation_python/model_file/shape_predictor_68_face_landmarks.dat")

        lmarks = []

        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))
        shapes = []
        for k, det in enumerate(dets):
            shape = predictor(img, det)
            shapes.append(shape)
            xy = _shape_to_np(shape)
            lmarks.append(xy)

        lmarks = np.asarray(lmarks, dtype='float32')

        # display_landmarks(img, dets, shapes)
        if (len(dets) > 0):
            # left - x1 top - y1 x2 - right y2 - bottom
            face_image = img[det.top():det.bottom(), det.left():det.right()]
            # cv2.imshow("Face Image",face_image)
            # cv2.waitKey(0)
            blank_image = np.ones((img.shape[0], img.shape[1], 3), np.uint8)
            blank_image[:, :] = 255
            blank_image[det.top():det.bottom(), det.left():det.right()] = face_image
            if (flag == 0):
                cv2.rectangle(img, (det.left(), det.top()), (det.right(), det.bottom()), (255, 0, 0), 1, 0, 0)
            # cv2.imshow("Face Detected Image", img)
            #cv2.imshow("White Background", blank_image)
            # cv2.waitKey(0)
        if (flag == 0):
            return len(dets), lmarks, blank_image
        else:
            return len(dets), lmarks



def display_landmarks(img, dets, shapes):
    win = dlib.image_window()
    win.clear_overlay()
    win.set_image(img)
    for shape in shapes:
        win.add_overlay(shape)
    win.add_overlay(dets)
    dlib.hit_enter_to_continue()