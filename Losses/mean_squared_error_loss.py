from typing import List
from .loss_abstract import Loss

class MeanSquaredError(Loss):
    def __init__(self) -> None:
        self.__name = "MeanSquaredErrorLoss"

    def compute(self, output_true: List[float], output_predicted: List[float]) -> List[float]:
        count = len(output_true)
        
        losses = []
        for (truth, predicted) in zip(output_true, output_predicted):
            losses.append(((predicted - truth) ** 2) / count)

        return losses

    def __str__(self):
        return self.__name