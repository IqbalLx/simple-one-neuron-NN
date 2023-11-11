from ActivationFunctions.relu_activation_function import ReLU
from Layers.dense_layer import Dense
from Losses.squared_error_loss import SquaredError
from Optimizers.gradient_descent_optimizer import GradientDescent
from Model.model import Model

import numpy as np

if __name__ == '__main__':
    x = [np.random.random_integers(1, 10) for _ in range(1, 100)]
    y = [2*i + np.random.random() for i in x]

    model = Model()
    layer = Dense(activation_function=ReLU())
    model.define(layer)

    print("Before training, layer params: ")
    print(layer)

    model.compile(loss=SquaredError(), optimizer=GradientDescent(learning_rate=0.00001))

    model.train(50, inputs=x, outputs=y)

    print("After training, layer params: ")
    print(layer)

    for ins, out in zip(x[:10], y[:10]):
        print(f"X: {ins}, Y: {out}, Y^: {layer.forward(input=ins)}")
