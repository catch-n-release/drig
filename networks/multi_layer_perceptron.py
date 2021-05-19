from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Activation


class MultiLayerPerceptron:
    @staticmethod
    def compose(nodes, regressor=False):
        net = Sequential()
        net.add(Dense(
            8,
            input_dim=nodes,
        ))
        net.add(Activation("relu"))
        net.add(Dense(4))
        net.add(Activation("relu"))

        if regressor:
            net.add(Dense(1))
            net.add(Activation("linear"))
        return net
