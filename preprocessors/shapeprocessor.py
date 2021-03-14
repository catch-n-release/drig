import cv2


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
            return cv2.resize(image, (self.width, self.height))
        except Exception as e:
            raise e
