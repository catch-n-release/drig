import pytest
from drig._tests.conftest import config
from drig.config import Error
from drig.utils import display_image, grab_image_cast, grab_image_paths, grab_image_class, grab_random_image, encode_class_names, grab_image_class_names
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder


def test_display_image_invalid_path(invalid_image_path):
    with pytest.raises(
            OSError, match=f"{Error.IMAGE_PATH_ERROR} : {invalid_image_path}"):
        display_image(invalid_image_path)


def test_grab_image_cast(random_testing_image_path):
    assert grab_image_cast(random_testing_image_path) == cv2.imread(
        random_testing_image_path).shape


def test_grab_image_cast_invalid_path(invalid_image_path):
    with pytest.raises(
            OSError, match=f"{Error.IMAGE_PATH_ERROR} : {invalid_image_path}"):
        grab_image_cast(invalid_image_path)


def test_grab_image_paths():
    all_image_paths = grab_image_paths(config.DATASET_PATH)
    assert all_image_paths
    assert type(all_image_paths) == list


def test_grab_image_paths_invalid_path(invalid_image_path):
    with pytest.raises(
            OSError,
            match=f"{Error.DATASET_PATH_ERROR} : {invalid_image_path}"):
        grab_image_paths(invalid_image_path)


def test_grab_image_paths_empty_dir(test_dir):
    with pytest.raises(Exception,
                       match=f"{Error.EMPTY_DATASET_ERROR} : {test_dir}"):
        grab_image_paths(test_dir)


def test_grab_image_class(random_testing_image_path):
    true_class_name = random_testing_image_path.split("/")[-2]
    class_name = grab_image_class(random_testing_image_path, -3)
    assert true_class_name == class_name


def test_grab_random_image_from_dataset():

    (
        image,
        class_name,
        image_path,
    ) = grab_random_image(
        dataset_path=config.DATASET_PATH,
        class_index=-3,
        return_image_path=True,
    )
    true_class_name = image_path.split("/")[-2]

    assert (
        type(image),
        class_name,
    ) == (
        np.ndarray,
        true_class_name,
    )


def test_grab_random_image_from_image_paths():
    all_image_paths = grab_image_paths(config.DATASET_PATH)
    (
        image,
        class_name,
        image_path,
    ) = grab_random_image(
        image_paths=all_image_paths,
        class_index=-3,
        return_image_path=True,
    )
    true_class_name = image_path.split("/")[-2]

    assert (
        type(image),
        class_name,
    ) == (
        np.ndarray,
        true_class_name,
    )


def test_encode_class_names():
    true_encoded_class_names = np.array([
        0,
        1,
        2,
    ])
    class_names = np.array([
        "cats",
        "dogs",
        "panda",
    ])
    assert np.array_equiv(
        np.sort(encode_class_names(class_names), axis=0),
        np.sort(true_encoded_class_names, axis=0),
    )


def test_grab_image_class_names_from_dataset():
    true_class_names = np.array([
        "cats",
        "dogs",
        "panda",
    ])

    class_names = grab_image_class_names(
        dataset_path=config.DATASET_PATH,
        class_index=-3,
    )

    assert np.array_equiv(
        np.sort(true_class_names, axis=0),
        np.sort(class_names, axis=0),
    )
