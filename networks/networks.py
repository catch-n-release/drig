from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras import backend
from keras.regularizers import l2
from keras.layers import concatenate, Input


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
            # CHUNK 1
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
            # CHUNK 2
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
            # CHUNK 3
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
            # CHUNK 4
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
                         padding="same"):
        try:
            tensor = Conv2D(filters, kernel, strides=strides,
                            padding=padding)(inputs)
            tensor = BatchNormalization(axis=channel_index)(tensor)
            tensor = Activation("relu")(tensor)
            return tensor
        except Exception as e:
            raise e

    @staticmethod
    def miniception_slab(inputs, conv_1_filters: int, conv_3_filters: int,
                         channel_index: int):
        try:
            strides = (1, 1)
            conv_1_slab = TinyGoogLeNet.convolution_slab(
                inputs, conv_1_filters, (1, 1), strides, channel_index)
            conv_3_slab = TinyGoogLeNet.convolution_slab(
                inputs, conv_3_filters, (3, 3), strides, channel_index)
            tensor = concatenate([conv_1_slab, conv_3_slab],
                                 axis=channel_index)
            return tensor
        except Exception as e:
            raise e

    @staticmethod
    def downsample_slab(inputs, filters: int, channel_index):
        try:
            conv_slab = TinyGoogLeNet.convolution_slab(inputs,
                                                       filters, (3, 3), (2, 2),
                                                       channel_index,
                                                       padding="valid")
            max_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(inputs)

            tensor = concatenate([conv_slab, max_pool], axis=channel_index)
            return tensor
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

            CHUNK 1

            """

            tensor = TinyGoogLeNet.convolution_slab(inputs, 96, (3, 3), (1, 1),
                                                    channel_index)
            """

            CHUNK 2

            """
            tensor = TinyGoogLeNet.miniception_slab(tensor, 32, 32,
                                                    channel_index)
            tensor = TinyGoogLeNet.miniception_slab(tensor, 32, 48,
                                                    channel_index)
            tensor = TinyGoogLeNet.downsample_slab(tensor, 80, channel_index)
            """

            CHUNK 3

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

            CHUNK 4

            """
            tensor = TinyGoogLeNet.miniception_slab(tensor, 176, 160,
                                                    channel_index)
            tensor = TinyGoogLeNet.miniception_slab(tensor, 176, 160,
                                                    channel_index)

            tensor = AveragePooling2D((7, 7))(tensor)
            tensor = Dropout(0.5)(tensor)
            """

            CHUNK 5

            """
            tensor = Flatten()(tensor)
            tensor = Dense(classes)(tensor)
            tensor = Activation("softmax")(tensor)

            net = Model(inputs, tensor, name="TinyGoogLeNet")

            return net

        except Exception as e:
            raise e
