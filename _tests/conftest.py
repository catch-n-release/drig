import pytest
import numpy as np
from imutils import paths
import cv2
from drig.datum_config import AnimalsConfig as config


@pytest.fixture
def random_image_path():
    random_image_path = np.random.choice(
        list(paths.list_images(config.DATASET_PATH)))
    return random_image_path


@pytest.fixture
def random_image(random_image_path):
    return cv2.imread(random_image_path)
