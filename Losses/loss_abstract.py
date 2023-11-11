from abc import ABC, abstractmethod
from typing import List

from Layers.layer_abstract import Layer

class Loss(ABC):

    @abstractmethod
    def compute(self, output_true: List[float], output_predicted: List[float]) -> List[float]:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass