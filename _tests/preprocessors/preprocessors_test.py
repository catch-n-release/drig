from drig.preprocessors import ShapePreprocessor, UniformAspectPreprocessor


def test_shape_preprocessor(random_image):
    resize_height = 256
    resize_width = 256
    shape_preprocessor = ShapePreprocessor(resize_height, resize_width)
    resized_image = shape_preprocessor.preprocess(random_image)
    assert (resized_image.shape[0], resized_image.shape[1]) == (resize_height,
                                                                resize_width)


def test_uniform_aspect_preprocessor(random_image):
    resize_height = 256
    resize_width = 256
    uniform_aspect_preprocessor = UniformAspectPreprocessor(
        resize_height, resize_width)
    resized_image = uniform_aspect_preprocessor.preprocess(random_image)
    assert (resized_image.shape[0], resized_image.shape[1]) == (resize_height,
                                                                resize_width)
