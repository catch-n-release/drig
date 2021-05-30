import pytest
from drig._tests.conftest import config
from drig.config import Error
from drig.utils import display_image, image_cast, list_image_paths
import cv2


def test_display_image_invalid_path(invalid_image_path):
    with pytest.raises(
            OSError, match=f"{Error.IMAGE_PATH_ERROR} : {invalid_image_path}"):
        display_image(invalid_image_path)


def test_image_cast(random_image_path):
    assert image_cast(random_image_path) == cv2.imread(random_image_path).shape


def test_image_cast_invalid_path(invalid_image_path):
    with pytest.raises(
            OSError, match=f"{Error.IMAGE_PATH_ERROR} : {invalid_image_path}"):
        image_cast(invalid_image_path)


def test_list_image_paths():
    all_image_paths = list_image_paths(config.DATASET_PATH)
    assert all_image_paths
    assert type(all_image_paths) == list


def test_list_image_paths_invalid_path(invalid_image_path):
    with pytest.raises(
            OSError,
            match=f"{Error.DATASET_PATH_ERROR} : {invalid_image_path}"):
        list_image_paths(invalid_image_path)


def test_list_image_paths_empty_dir(test_dir):
    with pytest.raises(Exception,
                       match=f"{Error.EMPTY_DATASET_ERROR} : {test_dir}"):
        list_image_paths(test_dir)
