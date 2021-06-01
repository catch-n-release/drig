from keras.regularizers import l2
from keras import backend
from keras.models import Model
from keras.layers import Input, add
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Flatten, Dense
from keras.layers.convolutional import Conv2D, AveragePooling2D
from drig.config import Kernel, Padding, PoolSize, Trigger, Stride


class ResNet:
    @staticmethod
    def residual_slab(
        influx: object,
        filters: int,
        channel_index: int,
        conv_stride: tuple = Stride.MESH_1x1,
        batch_norm_eps: float = 2e-5,
        batch_norm_mom: float = 9e-1,
        l2_notion: float = 1e-4,
        clip: bool = False,
    ):
        try:

            identity_tensor = influx

            conv_1n2_filters = int(filters * 0.25)

            ########################################

            batch_norm_1_tensor = BatchNormalization(
                axis=channel_index,
                epsilon=batch_norm_eps,
                momentum=batch_norm_mom,
            )(influx)

            trigger_1_tensor = Activation(Trigger.RELU)(batch_norm_1_tensor)

            conv_1_tensor = Conv2D(
                conv_1n2_filters,
                Kernel.MESH_1x1,
                use_bias=False,
                kernel_regularizer=l2(l2_notion),
            )(trigger_1_tensor)

            ########################################

            batch_norm_2_tensor = BatchNormalization(
                axis=channel_index,
                epsilon=batch_norm_eps,
                momentum=batch_norm_mom,
            )(conv_1_tensor)

            trigger_2_tensor = Activation(Trigger.RELU)(batch_norm_2_tensor)

            conv_2_tensor = Conv2D(
                conv_1n2_filters,
                Kernel.MESH_3x3,
                strides=conv_stride,
                padding=Padding.SAME,
                use_bias=False,
                kernel_regularizer=l2(l2_notion),
            )(trigger_2_tensor)

            ########################################

            batch_norm_3_tensor = BatchNormalization(
                axis=channel_index,
                epsilon=batch_norm_eps,
                momentum=batch_norm_mom,
            )(conv_2_tensor)

            trigger_3_tensor = Activation(Trigger.RELU)(batch_norm_3_tensor)

            conv_3_tensor = Conv2D(
                filters,
                Kernel.MESH_1x1,
                use_bias=False,
                kernel_regularizer=l2(l2_notion),
            )(trigger_3_tensor)

            #####################################

            if clip:
                identity_tensor = Conv2D(
                    filters,
                    Kernel.MESH_1x1,
                    strides=conv_stride,
                    use_bias=False,
                    kernel_regularizer=l2(l2_notion),
                )(trigger_1_tensor)

            residual_tensor = add([
                conv_3_tensor,
                identity_tensor,
            ])

            return residual_tensor

        except Exception as e:
            raise e

    @staticmethod
    def compose(
        height: int,
        width: int,
        depth: int,
        classes: int,
        config: dict,
        batch_norm_eps: float = 2e-5,
        batch_norm_mom: float = 9e-1,
        l2_notion: float = 1e-4,
        genre: str = "cifar",
    ):
        try:
            image_cast = (
                height,
                width,
                depth,
            )
            channel_index = -1

            if backend.image_data_format() == "channels_first":
                image_cast = (
                    depth,
                    height,
                    width,
                )
                channel_index = 1

            # slabs, filters = list(zip(*list(config.values())))

            influx = Input(shape=image_cast)

            ######################################

            tensor = BatchNormalization(
                axis=channel_index,
                epsilon=batch_norm_eps,
                momentum=batch_norm_mom,
            )(influx)

            if genre == "cifar":
                tensor = Conv2D(
                    config[0][1],
                    Kernel.MESH_3x3,
                    use_bias=False,
                    padding=Padding.SAME,
                    kernel_regularizer=l2(l2_notion),
                )(tensor)

            #######################################

            for step, (
                    slabs,
                    filters,
            ) in config.items():
                if step == 0:
                    continue
                conv_stride = Stride.MESH_1x1 if step == 1 else Stride.MESH_2x2

                tensor = ResNet.residual_slab(
                    tensor,
                    filters,
                    channel_index,
                    conv_stride,
                    batch_norm_eps,
                    batch_norm_mom,
                    clip=True,
                )

                for _ in range(slabs - 1):
                    tensor = ResNet.residual_slab(
                        tensor,
                        filters,
                        channel_index,
                        batch_norm_eps=batch_norm_eps,
                        batch_norm_mom=batch_norm_mom,
                        l2_notion=l2_notion,
                    )
            #########################################

            tensor = BatchNormalization(
                axis=channel_index,
                epsilon=batch_norm_eps,
                momentum=batch_norm_mom,
            )(tensor)

            tensor = Activation(Trigger.RELU)(tensor)

            tensor = AveragePooling2D(PoolSize.MESH_8x8)(tensor)

            #########################################

            tensor = Flatten()(tensor)

            tensor = Dense(
                classes,
                kernel_regularizer=l2(l2_notion),
            )(tensor)

            efflux = Activation(Trigger.SOFTMAX)(tensor)

            #########################################

            net = Model(influx, efflux, name="res_net")

            return net

        except Exception as e:
            raise e
