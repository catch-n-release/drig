import pytest
from drig.utils import display_image, image_cast
import cv2


def test_display_image_invalid_path():
    with pytest.raises(TypeError, match="INVALID IMAGE PATH"):
        display_image("random_image_path")


def test_image_cast(random_image_path):
    assert image_cast(random_image_path) == cv2.imread(random_image_path).shape


def test_image_cast_invalid_path():
    with pytest.raises(TypeError, match="INVALID IMAGE PATH"):
        image_cast("random_image_path")
