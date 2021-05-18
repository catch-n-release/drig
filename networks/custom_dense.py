from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras import backend
from keras.regularizers import l2
from keras.layers import concatenate, Input
from drig.config import Kernel, PoolSize, Stride, Padding


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
