from keras.layers.core import Flatten, Dense, Dropout


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
