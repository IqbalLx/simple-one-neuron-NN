from ActivationFunctions.activation_function_abstract import ActivationFunction
from .layer_abstract import Layer

class Dense(Layer):
    def __init__(self, activation_function: ActivationFunction = None) -> None:
        super().__init__(activation_function)

        self.__name = "DenseLayer"

        self.__weight = 1
        self.__bias = 0

    def forward(self, input: float) -> float:
        output = (self.__weight * input) + self.__bias

        if (self._activation_function is not None):
            return self._activation_function.apply(output)

        return output

    def update(self, **kwargs) -> None:
        self.__weight -= kwargs.get("weight_step_size")
        self.__bias -= kwargs.get("bias_step_size")

    def name(self) -> str:
        return self.__name
    
    def __str__(self) -> str:
        return f"y = {self.__weight}x + {self.__bias}"