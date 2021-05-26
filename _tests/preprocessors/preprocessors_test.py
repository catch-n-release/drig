from drig.preprocessors import *


def test_shape_preprocessor(random_image):
    resize_height = 256
    resize_width = 256
    shape_preprocessor = ShapePreprocessor(resize_height, resize_width)
    resized_image = shape_preprocessor.preprocess(random_image)
    assert resized_image.shape[0] == resize_height
    assert resized_image.shape[1] == resize_width
