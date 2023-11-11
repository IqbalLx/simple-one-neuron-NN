from abc import ABC, abstractmethod
from typing import List

from Layers.layer_abstract import Layer

class Optimizer(ABC):
    @abstractmethod
    def backpropagate(self, inputs: List[float], losses: List[float], layer: Layer) -> None:
        pass