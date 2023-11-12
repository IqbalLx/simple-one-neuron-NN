from ActivationFunctions.relu_activation_function import ReLU
from Layers.dense_layer import Dense
from Losses import MeanSquaredError
from Optimizers.gradient_descent_optimizer import GradientDescent
import Metrics as metrics
from Model.model import Model

import numpy as np

if __name__ == '__main__':
    x = [np.random.randint(1, 10) for _ in range(1, 1000)]
    y = [6.7*i + np.random.random() for i in x]

    x_train, x_test = x[:900], x[900:]
    y_train, y_test = y[:900], y[900:]

    model = Model()
    layer = Dense(activation_function=ReLU())
    model.define(layer)

    print("Before training, layer params: ")
    print(layer)

    model.compile(
        loss=MeanSquaredError(), 
        optimizer=GradientDescent(learning_rate=0.0001),
        metrics=[
            metrics.MeanAbsoluteError(),
            metrics.MeanSquaredError()
        ]
    )

    model.train(
        iterations=500, 
        inputs_train=x_train, 
        outputs_train=y_train, 
        inputs_test=x_test, 
        outputs_test=y_test
    )

    print("\nAfter training, layer params: ")
    print(layer)

    for ins, out in zip(x[:10], y[:10]):
        print(f"X: {ins}, Y: {out}, Y^: {model.predict(input=ins)}")
