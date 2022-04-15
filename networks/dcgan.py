from keras.models import Sequential
from keras.optimizer_v2.adam import Adam
from keras.layers import LeakyReLU, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Dropout, Flatten, Dense
from drig.config import ImageCast, Stride, Trigger, Padding, Kernel, Loss, Metrics
import numpy as np


class DCGAN:

    @staticmethod
    def discriminator(input_cast=ImageCast.GRAY_28x28):
        try:
            discriminator_net = Sequential()
            discriminator_net.add(
                Conv2D(
                    filters=64,
                    kernel_size=Kernel.MESH_3x3,
                    strides=Stride.MESH_2x2,
                    padding=Padding.SAME,
                    input_shape=input_cast,
                ))
            discriminator_net.add(LeakyReLU(alpha=0.2))
            discriminator_net.add(Dropout(0.4))
            discriminator_net.add(
                Conv2D(
                    filters=64,
                    kernel_size=Kernel.MESH_3x3,
                    strides=Stride.MESH_2x2,
                    padding=Padding.SAME,
                ))
            discriminator_net.add(LeakyReLU(alpha=0.2))
            discriminator_net.add(Dropout(0.4))
            discriminator_net.add(Flatten())
            discriminator_net.add(Dense(
                1,
                activation=Trigger.SIGMOID,
            ))
            refiner = Adam(learning_rate=0.0002, beta_1=0.5)
            discriminator_net.compile(
                loss=Loss.BI_CROSS,
                optimizer=refiner,
                metrics=Metrics.ACCURACY,
            )
            return discriminator_net

        except Exception as e:
            raise e

    @staticmethod
    def generator(
        embedding_cast: int,
        feature_maps: int = 128,
        image_cast: tuple() = ImageCast.GRAY_7x7,
    ):
        try:
            generator_net = Sequential()
            generator_net.add(
                Dense(
                    units=feature_maps * np.prod(image_cast),
                    input_dim=embedding_cast,
                ))
            generator_net.add(LeakyReLU(alpha=0.2))
            generator_net.add(
                Reshape((
                    image_cast[0],
                    image_cast[1],
                    feature_maps,
                )))

            generator_net.add(
                Conv2DTranspose(
                    filters=feature_maps,
                    kernel_size=Kernel.MESH_4x4,
                    strides=Stride.MESH_2x2,
                    padding=Padding.SAME,
                ))
            generator_net.add(LeakyReLU(alpha=0.2))
            generator_net.add(
                Conv2DTranspose(
                    filters=feature_maps,
                    kernel_size=Kernel.MESH_4x4,
                    strides=Stride.MESH_2x2,
                    padding=Padding.SAME,
                ))
            generator_net.add(LeakyReLU(alpha=0.2))
            generator_net.add(
                Conv2D(
                    filters=1,
                    kernel_size=Kernel.MESH_7x7,
                    activation=Trigger.SIGMOID,
                    padding=Padding.SAME,
                ))
            return generator_net
        except Exception as e:
            raise e

    @staticmethod
    def compose(generator_net, discriminator_net):
        try:
            discriminator_net.trainable = False
            dcgan = Sequential()
            dcgan.add(generator_net)
            dcgan.add(discriminator_net)
            refiner = Adam(learning_rate=0.0002, beta_1=0.5)
            dcgan.compile(loss=Loss.BI_CROSS, optimizer=refiner)
            return dcgan
        except Exception as e:
            raise e
