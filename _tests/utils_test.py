import pytest
from drig.utils import display_image


def test_display_image_invalid_path():
    with pytest.raises(TypeError, match="INVALID IMAGE PATH"):
        display_image("random_image_path")
