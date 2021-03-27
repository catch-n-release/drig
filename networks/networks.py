from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras import backend


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
