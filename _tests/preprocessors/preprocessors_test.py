from drig.preprocessors import ShapePreprocessor, UniformAspectPreprocessor, NormalizationPreprocessor, ImageToArrayPreprocessor, WindowPreprocessor, OverSamplingPreprocessor
import cv2
import numpy as np
from drig.config import ImageCast


def test_shape_preprocessor(random_testing_image):
    shape_preprocessor = ShapePreprocessor(
        ImageCast.RGB_256x256[0],
        ImageCast.RGB_256x256[1],
    )
    resized_image = shape_preprocessor.preprocess(random_testing_image)
    assert resized_image.shape == ImageCast.RGB_256x256


def test_uniform_aspect_preprocessor(random_testing_image):
    uniform_aspect_preprocessor = UniformAspectPreprocessor(
        ImageCast.RGB_256x256[0],
        ImageCast.RGB_256x256[1],
    )
    resized_image = uniform_aspect_preprocessor.preprocess(
        random_testing_image)
    assert resized_image.shape == ImageCast.RGB_256x256


def test_normalization_preprocessor(random_testing_image):
    mean_red, mean_green, mean_blue = 20, 30, 40
    blue_channel, green_channel, red_channel = cv2.split(
        random_testing_image.astype("float32"))
    true_red = red_channel - mean_red
    true_green = green_channel - mean_green
    true_blue = blue_channel - mean_blue
    norm_preprocessor = NormalizationPreprocessor(
        mean_red,
        mean_green,
        mean_blue,
    )
    normalized_image = norm_preprocessor.preprocess(random_testing_image)
    mean_red, mean_green, mean_blue = 20, 30, 40
    (
        normalized_blue_channel,
        normalized_green_channel,
        normalized_red_channel,
    ) = cv2.split(normalized_image.astype("float32"))

    assert ((true_red == normalized_red_channel).any()
            and (true_green == normalized_green_channel).any()
            and (true_blue == normalized_blue_channel).any())


def test_image_to_array_preprocessor(random_testing_image):
    image_array_preprocessor = ImageToArrayPreprocessor()
    image_array = image_array_preprocessor.preprocess(random_testing_image)
    assert type(image_array) == np.ndarray
    assert random_testing_image.shape == image_array.shape


def test_image_to_array_preprocessor_channels_first(random_testing_image):
    *random_image_dims, depth = random_testing_image.shape
    image_array_preprocessor = ImageToArrayPreprocessor(
        data_format="channels_first")
    image_array = image_array_preprocessor.preprocess(random_testing_image)
    assert type(image_array) == np.ndarray
    assert (
        depth,
        *random_image_dims,
    ) == image_array.shape


def test_window_preprocessor(random_testing_image):
    window_preprocessor = WindowPreprocessor(
        height=ImageCast.RGB_64x64[0],
        width=ImageCast.RGB_64x64[1],
    )
    image_patch = window_preprocessor.preprocess(random_testing_image)
    assert image_patch.shape == ImageCast.RGB_64x64


def test_over_sampling_preprocessor(random_testing_image):
    over_sample_preproc = OverSamplingPreprocessor(
        height=ImageCast.RGB_64x64[0],
        width=ImageCast.RGB_64x64[1],
    )
    over_sampled_images = over_sample_preproc.preprocess(random_testing_image)

    assert len(over_sampled_images) == 10
    assert all(
        map(
            lambda image: image.shape == ImageCast.RGB_64x64,
            over_sampled_images,
        ))
