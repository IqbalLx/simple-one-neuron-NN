from typing import List
from Layers.layer_abstract import Layer
from Losses.loss_abstract import Loss
from Metrics.metric_abstract import Metric
from Optimizers.optimizer_abtsract import Optimizer
from .model_abstract import Model as BaseModel

class Model(BaseModel):
    def __init__(self) -> None:
        super().__init__()

    def define(self, layer: Layer) -> None:
        self.__layer = layer

    def compile(self, loss: Loss, optimizer: Optimizer, metrics: List[Metric] = []) -> None:
        self.__loss = loss
        self.__optimizer = optimizer
        self.__metrics = metrics

    def __test(self, inputs_test: List[float], outputs_test: List[float]) -> None:
        outputs_predicted = [self.__layer.forward(i) for i in inputs_test]

        for metric in self.__metrics:
            value = metric.compute(output_truth=outputs_test, output_predicted=outputs_predicted)
            print(f"{metric.name()}Test: {value}")

    def train(self, 
              iterations: int, 
              inputs_train: List[float], 
              outputs_train: List[float],
              inputs_test: List[float] = [], 
              outputs_test: List[float] = []
              ) -> None:
        is_with_test = len(self.__metrics) > 0 and len(inputs_test) > 0 and len(inputs_test) == len(outputs_test)
        for i in range(0, iterations):
            print(f"\nBegin Iteration {i+1} ...\n")

            outputs_predicted = [self.__layer.forward(i) for i in inputs_train]
            losses = self.__loss.compute(outputs_train, outputs_predicted)
            
            print(f"Loss: {sum(losses)}")

            if (is_with_test): self.__test(inputs_test, outputs_test)
                
            self.__optimizer.backpropagate(inputs_train, losses, self.__layer)

    def predict(self, *args, **kwargs):
        return self.__layer.forward(*args, **kwargs)
