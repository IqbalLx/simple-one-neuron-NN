from abc import ABC, abstractmethod
from typing import List

from Layers.layer_abstract import Layer
from Losses.loss_abstract import Loss
from Metrics.metric_abstract import Metric
from Optimizers.optimizer_abtsract import Optimizer

class Model(ABC):

    @abstractmethod
    def define(self, layer: Layer) -> None:
        pass

    @abstractmethod
    def compile(self, loss: Loss, optimizer: Optimizer, metrics: List[Metric] = []) -> None:
        pass

    @abstractmethod
    def train(self, iterations: int, inputs: List[float], outputs: List[float]) -> None:
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass
