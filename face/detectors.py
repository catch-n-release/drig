import cv2
import numpy as np
import dlib
import cvlib as cv
import face_recognition
import os


class Detectors:
    def __init__(self, image):
        try:
            self.alt2_cascade_xml_path = os.path.abspath(
                os.path.join(
                    os.path.pardir,
                    "face/haar_cascades/haarcascade_frontalface_alt2.xml"))
            self.default_cascade_xml_path = os.path.abspath(
                os.path.join(
                    os.path.pardir,
                    "face/haar_cascades/haarcascade_frontalface_default.xml"))
            self.rgb_image = image
            self.gray_scale_image = cv2.cvtColor(self.rgb_image,
                                                 cv2.COLOR_BGR2GRAY)

        except Exception as e:
            raise e

    def haar_detector(self,
                      cascade_type="alt2",
                      scale=None,
                      min_size=None,
                      max_size=None):
        try:
            if cascade_type != "alt2":
                cascade = self.default_cascade_xml_path
            else:
                cascade = self.alt2_cascade_xml_path

            if scale or min_size or max_size:
                faces = cv2.face.getFacesHAAR(self.rgb_image, cascade)
                if faces:
                    faces = np.squeeze(faces)

            else:

                faces = cv2.CascadeClassifier(cascade).detectMultiScale(
                    self.gray_scale_image,
                    scaleFactor=scale,
                    minSize=min_size,
                    maxSize=max_size)
            faces = [[x, y, x + w, y + h] for x, y, w, h in faces]
            return np.array(faces)

        except Exception as e:
            raise e

    def dlib_hog_detector(self, scale=0):
        try:
            dlib_detector = dlib.get_frontal_face_detector()
            faces = dlib_detector(self.gray_scale_image, scale)
            faces = [[face.left(),
                      face.top(),
                      face.right(),
                      face.bottom()] for face in faces]
            return np.array(faces)
        except Exception as e:
            raise e

    def face_recog_dectector(self, scale=0, model="hog"):
        try:
            faces = face_recognition.face_locations(self.rgb_image, scale,
                                                    model)
            faces = [[x1, y1, x2, y2] for y1, x1, y2, x2 in faces]

            return np.array(faces)
        except Exception as e:
            raise e

    def cvlib_detector(self, threshold=0.5, gpu=False):
        try:
            faces, confidences = cv.detect_face(self.rgb_image,
                                                threshold=0.5,
                                                enable_gpu=gpu)
            faces = [[x1, y1, x2, y2] for x1, y1, x2, y2 in faces]

            return np.array(faces)
        except Exception as e:
            raise e


print(
    os.path.abspath(
        os.path.join(os.path.pardir,
                     "face/haar_cascades/haarcascade_frontalface_alt2.xml")))
