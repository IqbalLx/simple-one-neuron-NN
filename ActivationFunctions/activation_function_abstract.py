from abc import ABC, abstractmethod

class ActivationFunction(ABC):

    @abstractmethod
    def apply(self, layer_output: float) -> float:
        pass