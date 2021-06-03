import pytest
from drig.networks import ResNet
from drig.config import ImageCast, Stride
from drig.networks import ResNet


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


def test_res_net_compose(res_net_config):

    classes = 10
    res_net = ResNet.compose(
        *ImageCast.RGB_32x32,
        classes,
        res_net_config,
    )

    assert tuple(
        res_net.compute_output_shape(input_shape=(
            None,
            *ImageCast.RGB_32x32,
        ))) == (
            None,
            classes,
        )


def test_res_net_conv_layers(res_net_config):

    total_conv_layers = 1
    for step, (slabs, filters) in res_net_config.items():
        if step == 0:
            continue
        clipping_residual = 1
        non_clipping_residuals = slabs - clipping_residual
        conv_layers = (clipping_residual * 4) + (non_clipping_residuals * 3)
        total_conv_layers += conv_layers

    classes = 10
    res_net = ResNet.compose(
        *ImageCast.RGB_32x32,
        classes,
        res_net_config,
    )

    assert len(
        list(
            filter(lambda layer: layer["class_name"] == "Conv2D",
                   res_net.get_config()["layers"]))) == total_conv_layers


def test_res_net_avgpool(res_net_config):

    classes = 10
    res_net = ResNet.compose(
        *ImageCast.RGB_32x32,
        classes,
        res_net_config,
    )
    for index, layer in enumerate(res_net.get_config()["layers"]):
        if layer["class_name"] == "AveragePooling2D":
            break

    assert res_net.layers[index].output_shape == (
        None,
        1,
        1,
        256,
    )
