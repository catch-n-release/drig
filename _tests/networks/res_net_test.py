import pytest
from drig.networks import ResNet
from drig.config import ImageCast, Stride


def test_residual_slab(batch_norm_tensor):
    filters = 64
    residual_slab = ResNet.residual_slab(
        batch_norm_tensor,
        filters,
        channel_index=-1,
        clip=True,
    )
    _, *cast = residual_slab.shape.as_list()
    assert residual_slab.is_tensor_like
    assert tuple(cast) == (
        *ImageCast.RGB_64x64[:2],
        filters,
    )


def test_residual_slab_broadcasting(batch_norm_tensor):
    filters = 64
    with pytest.raises(ValueError):
        ResNet.residual_slab(
            batch_norm_tensor,
            filters,
            channel_index=-1,
        )


def test_residual_slab_cast_clip(batch_norm_tensor):
    filters = 64
    residual_slab = ResNet.residual_slab(
        batch_norm_tensor,
        filters,
        conv_stride=Stride.MESH_2x2,
        channel_index=-1,
        clip=True,
    )
    _, *cast = residual_slab.shape.as_list()
    assert residual_slab.is_tensor_like
    assert tuple(cast) == (
        *ImageCast.RGB_32x32[:2],
        filters,
    )
