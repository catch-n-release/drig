from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras import backend
from keras.regularizers import l2


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
