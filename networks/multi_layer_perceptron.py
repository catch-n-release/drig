from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Activation
from drig.config import Trigger


class MultiLayerPerceptron:
    @staticmethod
    def compose(nodes, regressor=False):
        net = Sequential()
        net.add(Dense(
            8,
            input_dim=nodes,
        ))
        net.add(Activation(Trigger.RELU))
        net.add(Dense(4))
        net.add(Activation(Trigger))

        if regressor:
            net.add(Dense(1))
            net.add(Activation(Trigger.LINEAR))
        return net
