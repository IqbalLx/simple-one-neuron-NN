from typing import List
from .metric_abstract import Metric

class MeanAbsoluteError(Metric):
    def __init__(self) -> None:
        super().__init__()

        self.__name = 'MeanAbsoluteErrorMetric'

    def compute(self, output_truth: List[float], output_predicted: List[float]) -> float:
        count = len(output_truth)
        mae = 0
        for (truth, predicted) in zip(output_truth, output_predicted):
            mae += (abs(predicted - truth)) / count

        return mae
    
    def name(self) -> str:
        return self.__name