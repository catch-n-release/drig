import cv2
import imutils
from keras.preprocessing.image import img_to_array
from sklearn.feature_extraction.image import extract_patches_2d


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


class NormalizationPreprocessor:
    def __init__(self, mean_red, mean_green, mean_blue):
        try:
            self.mean_red = mean_red
            self.mean_green = mean_green
            self.mean_blue = mean_blue
        except Exception as e:
            raise e

    def preprocess(self, image):
        try:
            blue_channel, green_channel, red_channel = cv2.split(image)
            blue_channel = blue_channel - self.mean_blue
            green_channel = green_channel - self.mean_green
            red_channel = red_channel - self.mean_red

            return cv2.merge([blue_channel, green_channel, red_channel])
        except Exception as e:
            raise e


class WindowPreprocessor:
    def __init__(self, width, height):
        try:
            self.width = width
            self.height = height
        except Exception as e:
            raise e

    def preprocess(self, image):
        try:
            return extract_patches_2d(image, (self.height, self.width),
                                      max_patches=1)[0]
        except Exception as e:
            raise e
