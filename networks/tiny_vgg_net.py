from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras import backend
from drig.config import Kernel, Trigger, PoolSize, Padding


class TinyVGGNet:

    @staticmethod
    def compose(
        height,
        width,
        depth,
        classes,
    ):
        try:
            input_cast = (
                height,
                width,
                depth,
            )
            channel_index = -1
            if backend.image_data_format() == "channels_first":
                input_cast = (
                    depth,
                    width,
                    height,
                )
                channel_index = 1

            net = Sequential()
            """

            SLAB 1

            """
            net.add(
                Conv2D(
                    32,
                    Kernel.MESH_3x3,
                    padding=Padding.SAME,
                    input_shape=input_cast,
                ))
            net.add(Activation(Trigger.RELU))
            net.add(BatchNormalization(axis=channel_index))

            net.add(Conv2D(
                32,
                Kernel.MESH_3x3,
                padding=Padding.SAME,
            ))
            net.add(Activation(Trigger.RELU))
            net.add(BatchNormalization(axis=channel_index))

            ############
            net.add(MaxPooling2D(pool_size=PoolSize.MESH_2x2))
            net.add(Dropout(0.25))
            ############
            """

            SLAB 2

            """
            net.add(Conv2D(
                64,
                Kernel.MESH_3x3,
                padding=Padding.SAME,
            ))
            net.add(Activation(Trigger.RELU))
            net.add(BatchNormalization(axis=channel_index))

            net.add(Conv2D(
                64,
                Kernel.MESH_3x3,
                padding=Padding.SAME,
            ))
            net.add(Activation(Trigger.RELU))
            net.add(BatchNormalization(axis=channel_index))

            #########
            net.add(MaxPooling2D(pool_size=PoolSize.MESH_2x2))
            net.add(Dropout(0.25))
            #########
            """

            SLAB 3

            """
            net.add(Flatten())
            net.add(Dense(512))
            net.add(Activation(Trigger.RELU))
            net.add(BatchNormalization())
            net.add(Dropout(0.50))
            """

            SLAB 4

            """
            net.add(Dense(classes))
            net.add(Activation(Trigger.SOFTMAX))
            return net

        except Exception as e:
            raise e
