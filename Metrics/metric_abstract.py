from abc import ABC, abstractmethod
from typing import List

class Metric(ABC):

    @abstractmethod
    def compute(self, output_truth: List[float], output_predicted: List[float]) -> float:
        pass

    @abstractmethod
    def name(self) -> str:
        pass