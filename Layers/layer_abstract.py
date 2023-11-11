from abc import ABC, abstractmethod

from ActivationFunctions.activation_function_abstract import ActivationFunction

class Layer(ABC):
    def __init__(self, activation_function: ActivationFunction = None) -> None:
        self._activation_function = activation_function

    @abstractmethod
    def forward(self, input: float) -> float:
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    
    @abstractmethod
    def __str__(self) -> str:
        pass