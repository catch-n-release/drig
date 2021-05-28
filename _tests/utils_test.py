import pytest
from drig._tests.conftest import config
from drig.utils import display_image, image_cast, list_image_paths
import cv2


def test_display_image_invalid_path():
    with pytest.raises(OSError, match="INVALID IMAGE PATH"):
        display_image("random_image_path")


def test_image_cast(random_image_path):
    assert image_cast(random_image_path) == cv2.imread(random_image_path).shape


def test_image_cast_invalid_path():
    with pytest.raises(OSError, match="INVALID IMAGE PATH"):
        image_cast("random_image_path")


def test_list_image_paths():
    all_image_paths = list_image_paths(config.DATASET_PATH)
    assert all_image_paths
    assert type(all_image_paths) == list


def test_list_image_paths_invalid_path():
    with pytest.raises(OSError, match="INVALID DATASET PATH"):
        list_image_paths("random_image_path")


def test_list_image_paths_empty_dir(test_dir):
    with pytest.raises(Exception, match="DIRECTORY HAS NO IMAGES"):
        list_image_paths(test_dir)
