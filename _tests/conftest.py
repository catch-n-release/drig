import pytest
import numpy as np
from imutils import paths
import cv2
from drig.datum_config import AnimalsConfig as config
import os
import shutil


@pytest.fixture
def random_testing_image_path():
    random_image_path = np.random.choice(
        list(paths.list_images(config.DATASET_PATH)))
    return random_image_path


@pytest.fixture
def random_testing_image(random_testing_image_path):
    return cv2.imread(random_testing_image_path)


@pytest.fixture
def test_dir():
    test_dir_path = os.path.join(os.pardir, "_tests/test_dir")
    os.makedirs(test_dir_path, exist_ok=True)
    yield test_dir_path
    shutil.rmtree(test_dir_path)


@pytest.fixture
def invalid_image_path():
    return "invalid/image/path.jpg"
