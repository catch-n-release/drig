from keras.models import Model
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras import backend
from keras.regularizers import l2
from keras.layers import concatenate, Input
from drig.config import Kernel, PoolSize, Stride, Padding


class NighGoogLeNet:
    @staticmethod
    def convolution_slab(
        influx,
        filters: int,
        kernel: tuple,
        stride: tuple,
        channel_index: int,
        padding: str = Padding.SAME,
        l2_norm: float = 0.0005,
        alias: str = None,
    ):
        try:
            (
                conv2d_alias,
                batch_norm_alias,
                activation_alias,
            ) = (
                None,
                None,
                None,
            )
            if not alias:
                conv2d_alias = f"{alias}_CONV"
                batch_norm_alias = f"{alias}_BATCH_NORM"
                activation_alias = f"{alias}_ACT"

            tensor = Conv2D(
                filters,
                kernel,
                strides=stride,
                padding=padding,
                kernel_regularizer=l2(l2_norm),
                name=conv2d_alias,
            )(influx)
            tensor = BatchNormalization(
                axis=channel_index,
                name=batch_norm_alias,
            )(tensor)
            tensor = Activation(
                "relu",
                name=activation_alias,
            )(tensor)
            return tensor
        except Exception as e:
            raise e

    @staticmethod
    def inception_slab(
        influx,
        primary_cascade_conv_1_filters: int,
        secondary_cascade_conv_1_filters: int,
        secondary_cascade_conv_3_filters: int,
        tertiary_cascade_conv_1_filters: int,
        tertiary_cascade_conv_5_filters: int,
        pooling_cascade_conv_1_filters: int,
        channel_index: int,
        alias: str,
        stride: tuple = Stride.MESH_1x1,
        l2_norm: float = 0.0005,
    ):
        try:

            primary_cascade = NighGoogLeNet.convolution_slab(
                influx,
                primary_cascade_conv_1_filters,
                Kernel.MESH_1x1,
                stride,
                channel_index=channel_index,
                l2_norm=l2_norm,
                alias=f"{alias}_primary_cascasde_conv_slab",
            )

            #######################

            secondary_tensor = NighGoogLeNet.convolution_slab(
                influx,
                secondary_cascade_conv_1_filters,
                Kernel.MESH_1x1,
                stride,
                channel_index=channel_index,
                l2_norm=l2_norm,
                alias=f"{alias}_secondary_cascasde_first_conv_slab",
            )

            secondary_cascade = NighGoogLeNet.convolution_slab(
                secondary_tensor,
                secondary_cascade_conv_3_filters,
                Kernel.MESH_3x3,
                stride,
                channel_index=channel_index,
                l2_norm=l2_norm,
                alias=f"{alias}_secondary_cascasde_second_conv_slab",
            )

            #########################

            tertiary_tensor = NighGoogLeNet.convolution_slab(
                influx,
                tertiary_cascade_conv_1_filters,
                Kernel.MESH_1x1,
                stride,
                channel_index=channel_index,
                l2_norm=l2_norm,
                alias=f"{alias}_tertiary_cascasde_first_conv_slab",
            )

            tertiary_cascade = NighGoogLeNet.convolution_slab(
                tertiary_tensor,
                tertiary_cascade_conv_5_filters,
                Kernel.MESH_5x5,
                stride,
                channel_index=channel_index,
                l2_norm=l2_norm,
                alias=f"{alias}_tertiary_cascasde_second_conv_slab",
            )

            ##########################

            pooling_tensor = MaxPooling2D(
                pool_size=PoolSize.MESH_3x3,
                strides=stride,
                padding=Padding.SAME,
                name=f"{alias}_pooling_cascade_max_pool_layer")(influx)
            pooling_cascade = NighGoogLeNet.convolution_slab(
                pooling_tensor,
                pooling_cascade_conv_1_filters,
                Kernel.MESH_1x1,
                stride,
                channel_index=channel_index,
                l2_norm=l2_norm,
                alias=f"{alias}_pooling_cascase_conv_slab",
            )

            inception_tensor = concatenate(
                [
                    primary_cascade, secondary_cascade, tertiary_cascade,
                    pooling_cascade
                ],
                axis=channel_index,
                name=f"{alias}_concat_layer",
            )

            return inception_tensor
        except Exception as e:
            raise e

    @staticmethod
    def compose(
        height: int,
        width: int,
        depth: int,
        classes: int,
        l2_norm: float = 0.0005,
    ):
        try:

            input_cast = (
                height,
                width,
                depth,
            )
            channel_index = -1

            if backend.image_data_format == "channels_first":
                input_cast = (
                    depth,
                    height,
                    width,
                )
                channel_index = 1
            ################################

            inputs = Input(shape=input_cast)
            '''

            SLAB 1

            '''

            tensor = NighGoogLeNet.convolution_slab(
                inputs,
                64,
                Kernel.MESH_5x5,
                Stride.MESH_1x1,
                channel_index,
                l2_norm=l2_norm,
                alias="slab_1",
            )

            tensor = MaxPooling2D(
                pool_size=PoolSize.MESH_3x3,
                strides=Stride.MESH_2x2,
                padding=Padding.SAME,
                name="max_pool_1",
            )(tensor)
            '''

            SLAB 2

            '''
            tensor = NighGoogLeNet.convolution_slab(
                tensor,
                64,
                Kernel.MESH_1x1,
                Stride.MESH_1x1,
                channel_index,
                l2_norm=l2_norm,
                alias="slab_2_conv_1",
            )

            tensor = NighGoogLeNet.convolution_slab(
                tensor,
                192,
                Kernel.MESH_3x3,
                Stride.MESH_1x1,
                channel_index,
                l2_norm=l2_norm,
                alias="slab_2_conv_2",
            )

            tensor = MaxPooling2D(
                pool_size=PoolSize.MESH_3x3,
                strides=Stride.MESH_2x2,
                padding=Padding.SAME,
                name="max_pool_2",
            )(tensor)
            '''

            SLAB 3

            '''
            tensor = NighGoogLeNet.inception_slab(
                tensor,
                64,
                96,
                128,
                16,
                32,
                32,
                channel_index,
                alias="slab_3_inception_1",
                l2_norm=l2_norm,
            )

            tensor = NighGoogLeNet.inception_slab(
                tensor,
                128,
                128,
                192,
                32,
                96,
                64,
                channel_index,
                alias="slab_3_inception_2",
                l2_norm=l2_norm,
            )

            tensor = MaxPooling2D(
                pool_size=PoolSize.MESH_3x3,
                strides=Stride.MESH_2x2,
                padding=Padding.SAME,
                name="max_pool_3",
            )(tensor)
            '''

            SLAB 4

            '''
            tensor = NighGoogLeNet.inception_slab(
                tensor,
                192,
                96,
                208,
                16,
                48,
                64,
                channel_index,
                alias="slab_4_inception_1",
                l2_norm=l2_norm,
            )

            tensor = NighGoogLeNet.inception_slab(
                tensor,
                160,
                112,
                224,
                24,
                64,
                64,
                channel_index,
                alias="slab_4_inception_2",
                l2_norm=l2_norm,
            )

            tensor = NighGoogLeNet.inception_slab(
                tensor,
                128,
                128,
                256,
                24,
                64,
                64,
                channel_index,
                alias="slab_4_inception_3",
                l2_norm=l2_norm,
            )

            tensor = NighGoogLeNet.inception_slab(
                tensor,
                112,
                144,
                288,
                32,
                64,
                64,
                channel_index,
                alias="slab_4_inception_4",
                l2_norm=l2_norm,
            )

            tensor = NighGoogLeNet.inception_slab(
                tensor,
                256,
                160,
                320,
                32,
                128,
                128,
                channel_index,
                alias="slab_4_inception_5",
                l2_norm=l2_norm,
            )

            tensor = MaxPooling2D(
                pool_size=PoolSize.MESH_3x3,
                strides=Stride.MESH_2x2,
                padding=Padding.SAME,
                name="max_pool_4",
            )(tensor)
            '''

            SLAB 5

            '''

            tensor = AveragePooling2D(
                pool_size=PoolSize.MESH_4x4,
                name="avg_pool_1",
            )(tensor)

            tensor = Dropout(
                0.4,
                name="dropout",
            )(tensor)

            tensor = Flatten(name="flatten", )(tensor)

            tensor = Dense(
                classes,
                name="classes",
            )(tensor)

            tensor = Activation(
                "softmax",
                name="softmax_act",
            )(tensor)

            net = Model(
                inputs,
                tensor,
                name="nigh_google_net",
            )

            #######################

            return net

        except Exception as e:
            raise e
