from drig.preprocessors import ShapePreprocessor, UniformAspectPreprocessor, NormalizationPreprocessor, ImageToArrayPreprocessor
import cv2
import numpy as np


def test_shape_preprocessor(random_image):
    resize_height = 256
    resize_width = 256
    shape_preprocessor = ShapePreprocessor(
        resize_height,
        resize_width,
    )
    resized_image = shape_preprocessor.preprocess(random_image)
    assert (
        resized_image.shape[0],
        resized_image.shape[1],
    ) == (
        resize_height,
        resize_width,
    )


def test_uniform_aspect_preprocessor(random_image):
    resize_height = 256
    resize_width = 256
    uniform_aspect_preprocessor = UniformAspectPreprocessor(
        resize_height, resize_width)
    resized_image = uniform_aspect_preprocessor.preprocess(random_image)
    assert resized_image.shape[:2] == (
        resize_height,
        resize_width,
    )


def test_normalization_preprocessor(random_image):
    mean_red, mean_green, mean_blue = 20, 30, 40
    blue_channel, green_channel, red_channel = cv2.split(
        random_image.astype("float32"))
    true_red = red_channel - mean_red
    true_green = green_channel - mean_green
    true_blue = blue_channel - mean_blue
    norm_preprocessor = NormalizationPreprocessor(
        mean_red,
        mean_green,
        mean_blue,
    )
    normalized_image = norm_preprocessor.preprocess(random_image)
    mean_red, mean_green, mean_blue = 20, 30, 40
    (
        normalized_blue_channel,
        normalized_green_channel,
        normalized_red_channel,
    ) = cv2.split(normalized_image.astype("float32"))

    assert ((true_red == normalized_red_channel).any()
            and (true_green == normalized_green_channel).any()
            and (true_blue == normalized_blue_channel).any())


def test_image_to_array_preprocessor(random_image):
    height, width, depth = random_image.shape
    image_array_preprocessor = ImageToArrayPreprocessor()
    image_array = image_array_preprocessor.preprocess(random_image)
    assert type(image_array) == np.ndarray
    assert (height, width, depth) == image_array.shape


def test_image_to_array_preprocessor_channels_first(random_image):
    height, width, depth = random_image.shape
    image_array_preprocessor = ImageToArrayPreprocessor(
        data_format="channels_first")
    image_array = image_array_preprocessor.preprocess(random_image)
    assert type(image_array) == np.ndarray
    assert (
        depth,
        height,
        width,
    ) == image_array.shape
