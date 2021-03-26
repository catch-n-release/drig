import cv2
import imutils
from keras.preprocessing.image import img_to_array


class ShapePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        try:
            self.width = width
            self.height = height
            self.inter = inter
        except Exception as e:
            raise e

    def preprocess(self, image):
        try:
            return cv2.resize(image, (self.width, self.height),
                              interpolation=self.inter)
        except Exception as e:
            raise e


class UniformAspectPreprocessor():
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        try:
            self.width = width
            self.height = height
            self.inter = inter
        except Exception as e:
            raise e

    def preprocess(self, image):
        try:
            h, w = image.shape[:2]
            delta_w = 0
            delta_h = 0

            if w < h:
                image = imutils.resize(image,
                                       width=self.width,
                                       inter=self.inter)
                delta_h = int((image.shape[0] - self.height) / 2.0)
            else:
                image = imutils.resize(image,
                                       height=self.height,
                                       inter=self.inter)
                delta_w = int((image.shape[1] - self.width) / 2.0)
            new_h, new_w = image.shape[:2]

            image = image[delta_h:new_h - delta_h, delta_w:new_w - delta_w]

            return cv2.resize(image, (self.width, self.height),
                              interpolation=self.inter)

        except Exception as e:
            raise e


class ImageToArrayPreprocessor:
    def __init__(self, data_format=None):
        try:
            self.data_format = data_format
        except Exception as e:
            raise e

    def preprocess(self, image):
        try:
            return img_to_array(image, data_format=self.data_format)
        except Exception as e:
            raise e
