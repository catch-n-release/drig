import cv2
import imutils
from keras.preprocessing.image import img_to_array
from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class ShapePreprocessor:

    def __init__(
        self,
        height: int,
        width: int,
        inter=cv2.INTER_AREA,
    ):
        try:
            self.width = width
            self.height = height
            self.inter = inter
        except Exception as e:
            raise e

    def preprocess(
        self,
        image: np.ndarray,
    ):
        try:
            return cv2.resize(
                image,
                (self.width, self.height),
                interpolation=self.inter,
            )
        except Exception as e:
            raise e


class UniformAspectPreprocessor():

    def __init__(
        self,
        height: int,
        width: int,
        inter=cv2.INTER_AREA,
    ):
        try:
            self.width = width
            self.height = height
            self.inter = inter
        except Exception as e:
            raise e

    def preprocess(
        self,
        image: np.ndarray,
    ):
        try:
            h, w = image.shape[:2]
            delta_w = 0
            delta_h = 0

            if w < h:
                image = imutils.resize(
                    image,
                    width=self.width,
                    inter=self.inter,
                )
                delta_h = int((image.shape[0] - self.height) / 2.0)
            else:
                image = imutils.resize(
                    image,
                    height=self.height,
                    inter=self.inter,
                )
                delta_w = int((image.shape[1] - self.width) / 2.0)
            new_h, new_w = image.shape[:2]

            image = image[delta_h:new_h - delta_h, delta_w:new_w - delta_w]

            return cv2.resize(
                image,
                (self.width, self.height),
                interpolation=self.inter,
            )

        except Exception as e:
            raise e


class ImageToArrayPreprocessor:

    def __init__(
        self,
        data_format: str = None,
    ):
        try:
            self.data_format = data_format
        except Exception as e:
            raise e

    def preprocess(
        self,
        image: np.ndarray,
    ):
        try:
            return img_to_array(
                image,
                data_format=self.data_format,
            )
        except Exception as e:
            raise e


class MeanNormalizationPreprocessor:

    def __init__(
        self,
        mean_red: float,
        mean_green: float,
        mean_blue: float,
    ):
        try:
            self.mean_red = mean_red
            self.mean_green = mean_green
            self.mean_blue = mean_blue
        except Exception as e:
            raise e

    def preprocess(
        self,
        image: np.ndarray,
    ):
        try:
            (blue_channel, green_channel,
             red_channel) = cv2.split(image.astype("float32"))
            blue_channel -= self.mean_blue
            green_channel -= self.mean_green
            red_channel -= self.mean_red

            return cv2.merge([blue_channel, green_channel, red_channel])
        except Exception as e:
            raise e


class WindowPreprocessor:

    def __init__(
        self,
        height: int,
        width: int,
    ):
        try:
            self.height = height
            self.width = width
        except Exception as e:
            raise e

    def preprocess(
        self,
        image: np.ndarray,
    ):
        try:
            return extract_patches_2d(image, (self.height, self.width),
                                      max_patches=1)[0]
        except Exception as e:
            raise e


class OverSamplingPreprocessor:

    def __init__(
        self,
        height: int,
        width: int,
        interpolation=cv2.INTER_AREA,
        x_flip: bool = True,
    ):
        try:
            self.height = height
            self.width = width
            self.interpolation = interpolation
            self.x_flip = x_flip
        except Exception as e:
            raise e

    def preprocess(
        self,
        image: np.ndarray,
    ):
        try:
            crops = list()
            image_height, image_width = image.shape[:2]
            top_left = (0, 0, self.width, self.height)
            top_right = (image_width - self.width, 0, image_width, self.height)
            bottom_left = (0, image_height - self.height, self.width,
                           image_height)
            bottom_right = (image_width - self.width,
                            image_height - self.height, image_width,
                            image_height)
            delta_width = int(0.5 * (image_width - self.width))
            delta_height = int(0.5 * (image_height - self.height))
            center = (delta_width, delta_height, image_width - delta_width,
                      image_height - delta_height)
            points = [top_left, top_right, bottom_left, bottom_right, center]

            for left, top, right, bottom, in points:
                cropped_image = image[top:bottom, left:right]

                cropped_image = cv2.resize(cropped_image,
                                           (self.width, self.height),
                                           interpolation=self.interpolation)
                crops.append(cropped_image)

            if self.x_flip:
                flipped_cropped_images = [
                    cv2.flip(cropped_image, 1) for cropped_image in crops
                ]
            crops.extend(flipped_cropped_images)

            return np.array(crops)
        except Exception as e:
            raise e


class NormalizationPreprocessor:

    def __init__(
            self,
            scale: tuple = (0, 1),
            to_float=True,
    ):
        try:
            self.to_float = to_float
            self.normalizer = MinMaxScaler(feature_range=scale)
        except Exception as e:
            raise e

    def preprocess(self, image: np.ndarray):
        try:
            image = image.astype("float32") if self.to_float else image
            return self.normalizer.fit_transform(image)
        except Exception as e:
            raise e
