from typing import List
from Layers.layer_abstract import Layer
from .optimizer_abtsract import Optimizer

class GradientDescent(Optimizer):
    def __init__(self, learning_rate=0.01) -> None:
        self.__learning_rate: float = learning_rate

        super().__init__()

    def __backpropagate_dense_layer(self, inputs: List[float], losses: List[float], layer: Layer) -> None:
        loss_wrt_weight_derivatives = sum([-2 * loss * input for input, loss in zip(inputs, losses)])
        loss_wrt_bias_derivatives = sum([-2 * loss for loss in losses])
        
        weight_step_size = loss_wrt_weight_derivatives * self.__learning_rate
        bias_step_size = loss_wrt_bias_derivatives * self.__learning_rate

        layer.update(weight_step_size=weight_step_size, bias_step_size=bias_step_size)

    def backpropagate(self, inputs: List[float], losses: List[float], layer: Layer) -> None:
        if (layer.name() == 'DenseLayer'): return self.__backpropagate_dense_layer(inputs, losses, layer)

        return
        