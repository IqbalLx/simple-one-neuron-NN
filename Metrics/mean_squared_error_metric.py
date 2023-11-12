from typing import List
from .metric_abstract import Metric

class MeanSquaredError(Metric):
    def __init__(self) -> None:
        super().__init__()

        self.__name = 'MeanSquaredErrorMetric'

    def compute(self, output_truth: List[float], output_predicted: List[float]) -> float:
        count = len(output_truth)
        mse = 0
        for (truth, predicted) in zip(output_truth, output_predicted):
            mse += ((predicted - truth) ** 2) / count

        return mse
    
    def name(self) -> str:
        return self.__name