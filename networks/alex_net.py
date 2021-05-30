from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras import backend
from keras.regularizers import l2
from drig.config import Kernel, PoolSize, Stride, Padding


class AlexNet:
    @staticmethod
    def compose(height, width, depth, classes, l2_regularization=0.0002):
        try:

            input_cast = (height, width, depth)
            channel_index = -1
            if backend.image_data_format() == "channels_first":
                input_cast = (
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
                Conv2D(
                    96,
                    Kernel.MESH_11x11,
                    strides=Stride.MESH_4x4,
                    padding=Padding.SAME,
                    input_shape=input_cast,
                    kernel_regularizer=l2(l2_regularization),
                ))
            net.add(Activation("relu"))
            net.add(BatchNormalization(axis=channel_index))
            net.add(
                MaxPooling2D(
                    pool_size=PoolSize.MESH_3x3,
                    strides=Stride.MESH_2x2,
                ))
            net.add(Dropout(0.25))
            ########
            # SLAB 2
            ########
            net.add(
                Conv2D(
                    256,
                    Kernel.MESH_5x5,
                    padding=Padding.SAME,
                    kernel_regularizer=l2(l2_regularization),
                ))
            net.add(Activation("relu"))
            net.add(BatchNormalization(axis=channel_index))
            net.add(
                MaxPooling2D(
                    pool_size=PoolSize.MESH_3x3,
                    strides=Stride.MESH_2x2,
                ))
            net.add(Dropout(0.25))
            #########
            # SLAB 3
            #########
            net.add(
                Conv2D(
                    384,
                    Kernel.MESH_3x3,
                    padding=Padding.SAME,
                    kernel_regularizer=l2(l2_regularization),
                ))
            net.add(Activation("relu"))
            net.add(BatchNormalization(axis=channel_index))
            ########
            net.add(
                Conv2D(
                    384,
                    Kernel.MESH_3x3,
                    padding=Padding.SAME,
                    kernel_regularizer=l2(l2_regularization),
                ))
            net.add(Activation("relu"))
            net.add(BatchNormalization(axis=channel_index))
            ########
            net.add(
                Conv2D(
                    256,
                    Kernel.MESH_3x3,
                    padding=Padding.SAME,
                    kernel_regularizer=l2(l2_regularization),
                ))
            net.add(Activation("relu"))
            net.add(BatchNormalization(axis=channel_index))
            ########
            net.add(
                MaxPooling2D(pool_size=PoolSize.MESH_3x3,
                             strides=Stride.MESH_2x2), )
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
