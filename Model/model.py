from typing import List
from Layers.layer_abstract import Layer
from Losses.loss_abstract import Loss
from Optimizers.optimizer_abtsract import Optimizer
from .model_abstract import Model as BaseModel

class Model(BaseModel):
    def __init__(self) -> None:
        super().__init__()

    def define(self, layer: Layer) -> None:
        self.__layer = layer

    def compile(self, loss: Loss, optimizer: Optimizer) -> None:
        self.__loss = loss
        self.__optimizer = optimizer

    def train(self, iterations: int, inputs: List[float], outputs: List[float]) -> None:
        for i in range(0, iterations):
            print(f"Begin Iteration {i+1} ...\n")

            outputs_predicted = [self.__layer.forward(i) for i in inputs]
            losses = self.__loss.compute(outputs, outputs_predicted)
            
            print(f"Loss: {sum(losses)}\n")

            self.__optimizer.backpropagate(inputs, losses, self.__layer)
