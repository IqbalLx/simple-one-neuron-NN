from typing import List
from .loss_abstract import Loss

class SquaredError(Loss):
    def __init__(self) -> None:
        self.__name = "SquaredErrorLoss"

    def compute(self, output_true: List[float], output_predicted: List[float]) -> List[float]:
        losses = []
        for (truth, predicted) in zip(output_true, output_predicted):
            losses.append((predicted - truth) ** 2)

        return losses

    def __str__(self):
        return self.__name