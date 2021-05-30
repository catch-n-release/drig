from keras.models import Input, Model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras import backend
from drig.config import Kernel, PoolSize, Padding, Trigger


class CustomNet:
    @staticmethod
    def compose(
            height: int,
            width: int,
            depth: int,
            classes: int = None,
            filters: list = (16, 32, 64),
            regressor: bool = False,
    ):
        try:
            input_cast = (height, width, depth)
            channel_index = -1

            if backend.image_data_format() == "channels_first":
                input_cast = (depth, height, width)
                channel_index = 1

            inputs = Input(shape=input_cast)

            tensor = inputs
            for filter_size in filters:
                tensor = Conv2D(
                    filter_size,
                    Kernel.MESH_3x3,
                    padding=Padding.SAME,
                )(tensor)
                tensor = Activation(Trigger.RELU)(tensor)
                tensor = BatchNormalization(axis=channel_index)(tensor)
                tensor = MaxPooling2D(pool_size=PoolSize.MESH_2x2)(tensor)

            ##################
            tensor = Flatten()(tensor)
            tensor = Dense(16)(tensor)
            tensor = Activation(Trigger.RELU)(tensor)
            tensor = BatchNormalization(axis=channel_index)(tensor)
            tensor = Dropout(0.5)(tensor)

            #################

            tensor = Dense(4)(tensor)
            tensor = Activation(Trigger.RELU)(tensor)

            ###############
            if regressor:
                tensor = Dense(1)(tensor)
                tensor = Activation(Trigger.LINEAR)(tensor)

            ###############

            net = Model(inputs, tensor)
            return net
        except Exception as e:
            raise e
