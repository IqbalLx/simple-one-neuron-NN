from .activation_function_abstract import ActivationFunction


class ReLU(ActivationFunction):

    def __init__(self) -> None:
        super().__init__()

    def apply(self, layer_output: float) -> float:
        return max(layer_output, 0)