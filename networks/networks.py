from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras import backend
from keras.regularizers import l2
from keras.layers import concatenate, Input
from drig.config import Kernel, PoolSize, Stride, Padding


class AlexNet:
    @staticmethod
    def compose(height, width, depth, classes, l2_regularization=0.0002):
        try:

            input_dim = (height, width, depth)
            channel_index = -1
            if backend.image_data_format() == "channels_first":
                input_dim = (
                    depth,
                    height,
                    width,
                )
                channel_index = 1
            net = Sequential()
            ########
            # SLAB 1
            ########
            net.add(
                Conv2D(96, (11, 11),
                       strides=(4, 4),
                       padding="same",
                       input_shape=input_dim,
                       kernel_regularizer=l2(l2_regularization)))
            net.add(Activation("relu"))
            net.add(BatchNormalization(axis=channel_index))
            net.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
            net.add(Dropout(0.25))
            ########
            # SLAB 2
            ########
            net.add(
                Conv2D(256, (5, 5),
                       padding="same",
                       kernel_regularizer=l2(l2_regularization)))
            net.add(Activation("relu"))
            net.add(BatchNormalization(axis=channel_index))
            net.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
            net.add(Dropout(0.25))
            #########
            # SLAB 3
            #########
            net.add(
                Conv2D(384, (3, 3),
                       padding="same",
                       kernel_regularizer=l2(l2_regularization)))
            net.add(Activation("relu"))
            net.add(BatchNormalization(axis=channel_index))
            ########
            net.add(
                Conv2D(384, (3, 3),
                       padding="same",
                       kernel_regularizer=l2(l2_regularization)))
            net.add(Activation("relu"))
            net.add(BatchNormalization(axis=channel_index))
            ########
            net.add(
                Conv2D(256, (3, 3),
                       padding="same",
                       kernel_regularizer=l2(l2_regularization)))
            net.add(Activation("relu"))
            net.add(BatchNormalization(axis=channel_index))
            ########
            net.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
            net.add(Dropout(0.25))
            ########
            # SLAB 4
            ########
            net.add(Flatten())
            net.add(Dense(4096, kernel_regularizer=l2(l2_regularization)))
            net.add(Activation("relu"))
            net.add(BatchNormalization())
            net.add(Dropout(0.5))
            #########
            net.add(Dense(4096, kernel_regularizer=l2(l2_regularization)))
            net.add(Activation("relu"))
            net.add(BatchNormalization())
            net.add(Dropout(0.5))
            #########
            net.add(Dense(classes, kernel_regularizer=l2(l2_regularization)))
            net.add(Activation("softmax"))

            return net
        except Exception as e:
            raise e


class TinyVGG:
    @staticmethod
    def build(width, height, depth, classes):
        try:
            input_dim = (height, width, depth)
            channel_index = -1
            if backend.image_data_format() == "channels_first":
                input_dim = (depth, width, height)
                channel_index = 1
            model = Sequential()
            ##########
            model.add(Conv2D(32, (3, 3), padding="same",
                             input_shape=input_dim))
            model.add(Activation("relu"))
            model.add(BatchNormalization(axis=channel_index))

            model.add(Conv2D(
                32,
                (3, 3),
                padding="same",
            ))
            model.add(Activation("relu"))
            model.add(BatchNormalization(axis=channel_index))
            ############
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            ############
            model.add(Conv2D(64, (3, 3), padding="same"))
            model.add(Activation("relu"))
            model.add(BatchNormalization(axis=channel_index))

            model.add(Conv2D(
                64,
                (3, 3),
                padding="same",
            ))
            model.add(Activation("relu"))
            model.add(BatchNormalization(axis=channel_index))
            #########
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            ###########
            model.add(Flatten())
            model.add(Dense(512))
            model.add(Activation("relu"))
            model.add(BatchNormalization())
            model.add(Dropout(0.50))
            #######
            model.add(Dense(classes))
            model.add(Activation("softmax"))
            return model

        except Exception as e:
            raise e


class CustomDense:
    @staticmethod
    def compose(base_network, nodes, classes):
        try:
            custom_dense = base_network.output
            ######
            custom_dense = Flatten(name="flatten")(custom_dense)
            ######
            custom_dense = Dense(nodes, activation="relu")(custom_dense)
            # custom_dense = Activation("relu")(custom_dense)
            custom_dense = Dropout(0.5)(custom_dense)
            custom_dense = Dense(classes, activation="softmax")(custom_dense)
            # custom_dense = Activation("softmax")(custom_dense)
            return custom_dense
        except Exception as e:
            raise e


class TinyGoogLeNet:
    @staticmethod
    def convolution_slab(inputs,
                         filters: int,
                         kernel: tuple,
                         strides: tuple,
                         channel_index: int,
                         padding: str = "same"):
        try:
            tensor = Conv2D(filters, kernel, strides=strides,
                            padding=padding)(inputs)
            tensor = BatchNormalization(axis=channel_index)(tensor)
            convolution_tensor = Activation("relu")(tensor)
            return convolution_tensor
        except Exception as e:
            raise e

    @staticmethod
    def miniception_slab(
            inputs,
            conv_1_filters: int,
            conv_3_filters: int,
            channel_index: int,
            strides: tuple = (1, 1),
    ):
        try:

            conv_1_slab = TinyGoogLeNet.convolution_slab(
                inputs, conv_1_filters, (1, 1), strides, channel_index)
            conv_3_slab = TinyGoogLeNet.convolution_slab(
                inputs, conv_3_filters, (3, 3), strides, channel_index)
            miniception_tensor = concatenate([conv_1_slab, conv_3_slab],
                                             axis=channel_index)
            return miniception_tensor
        except Exception as e:
            raise e

    @staticmethod
    def downsample_slab(inputs, filters: int, channel_index: int):
        try:
            conv_slab = TinyGoogLeNet.convolution_slab(inputs,
                                                       filters, (3, 3), (2, 2),
                                                       channel_index,
                                                       padding="valid")
            max_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(inputs)

            downsample_tensor = concatenate([conv_slab, max_pool],
                                            axis=channel_index)
            return downsample_tensor
        except Exception as e:
            raise e

    @staticmethod
    def compose(height: int, width: int, depth: int, classes: int):
        try:
            input_dim = (height, width, depth)
            channel_index = -1

            if backend.image_data_format == "channels_first":
                input_dim = (depth, height, width)
                channel_index = 1

            inputs = Input(shape=input_dim)
            """

            SLAB 1

            """

            tensor = TinyGoogLeNet.convolution_slab(inputs, 96, (3, 3), (1, 1),
                                                    channel_index)
            """

            SLAB 2

            """
            tensor = TinyGoogLeNet.miniception_slab(tensor, 32, 32,
                                                    channel_index)
            tensor = TinyGoogLeNet.miniception_slab(tensor, 32, 48,
                                                    channel_index)
            tensor = TinyGoogLeNet.downsample_slab(tensor, 80, channel_index)
            """

            SLAB 3

            """
            tensor = TinyGoogLeNet.miniception_slab(tensor, 112, 48,
                                                    channel_index)
            tensor = TinyGoogLeNet.miniception_slab(tensor, 96, 64,
                                                    channel_index)
            tensor = TinyGoogLeNet.miniception_slab(tensor, 80, 80,
                                                    channel_index)
            tensor = TinyGoogLeNet.miniception_slab(tensor, 48, 96,
                                                    channel_index)
            tensor = TinyGoogLeNet.downsample_slab(tensor, 96, channel_index)
            """

            SLAB 4

            """
            tensor = TinyGoogLeNet.miniception_slab(tensor, 176, 160,
                                                    channel_index)
            tensor = TinyGoogLeNet.miniception_slab(tensor, 176, 160,
                                                    channel_index)

            tensor = AveragePooling2D((7, 7))(tensor)
            tensor = Dropout(0.5)(tensor)
            """

            SLAB 5

            """
            tensor = Flatten()(tensor)
            tensor = Dense(classes)(tensor)
            tensor = Activation("softmax")(tensor)

            net = Model(inputs, tensor, name="tiny_google_net")

            return net

        except Exception as e:
            raise e


class NighGoogLeNet:
    @staticmethod
    def convolution_slab(
        inputs,
        filters: int,
        kernel: tuple,
        stride: tuple,
        channel_index: int,
        padding: str = Padding.SAME,
        l2_norm: float = 0.0005,
        alias: str = None,
    ):
        try:
            conv2d_alias, batch_norm_alias, activation_alias = None, None, None
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
            )(inputs)
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
        inputs,
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
                inputs,
                primary_cascade_conv_1_filters,
                Kernel.MESH_1x1,
                stride,
                channel_index=channel_index,
                l2_norm=l2_norm,
                alias=f"{alias}_primary_cascasde_conv_slab",
            )

            #######################

            secondary_tensor = NighGoogLeNet.convolution_slab(
                inputs,
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
                inputs,
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
                name=f"{alias}_pooling_cascade_max_pool_layer")(inputs)
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

            input_dim = (height, width, depth)
            channel_index = -1

            if backend.image_data_format == "channels_first":
                input_dim = (depth, height, width)
                channel_index = 1
            ################################

            inputs = Input(shape=input_dim)
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
