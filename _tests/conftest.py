import pytest
import numpy as np
from imutils import paths
import cv2
from drig.datum_config import AnimalsConfig as config
from drig.config import ImageCast
import os
import shutil
from keras.layers import Input
from keras.layers.normalization import BatchNormalization


@pytest.fixture
def random_image_path():
    random_image_path = np.random.choice(
        list(paths.list_images(config.DATASET_PATH)))
    return random_image_path


@pytest.fixture
def random_image(random_image_path):
    return cv2.imread(random_image_path)


@pytest.fixture
def test_dir():
    test_dir_path = os.path.join(os.pardir, "_tests/test_dir")
    os.makedirs(test_dir_path, exist_ok=True)
    yield test_dir_path
    shutil.rmtree(test_dir_path)


@pytest.fixture
def invalid_image_path():
    return "invalid/image/path.jpg"


@pytest.fixture
def batch_norm_tensor():
    influx = Input(shape=ImageCast.RGB_64x64)
    axis = -1
    batch_norm_tensor = BatchNormalization(
        axis=axis,
        epsilon=2e-5,
        momentum=9e-1,
    )(influx)
    return batch_norm_tensor


@pytest.fixture
def res_net_config():
    return {
        0: (0, 64),
        1: (9, 64),
        2: (9, 128),
        3: (9, 256),
    }
