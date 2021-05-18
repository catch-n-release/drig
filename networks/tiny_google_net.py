from keras.models import Model
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras import backend
from keras.layers import concatenate, Input


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
