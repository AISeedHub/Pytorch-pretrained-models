from torchmetrics.classification import Accuracy, Dice, F1Score
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics import MetricCollection
from typing import Dict, Union
import torch


class CustomMAP(MeanAveragePrecision):
    '''
    Customized MeanAveragePrecision for pytorch-lightning
    '''
    def __init__(self):
        super().__init__()

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        '''
        Update metric
        Args:
            preds: predicted output
            target: ground truth
        '''
        preds = [{k: v.cpu() for k, v in t.items()} for t in preds]
        target = [{k: v.cpu() for k, v in t.items()} for t in target]
        if 'scores' not in preds[0].keys():
            preds = [{**t, 'scores': torch.ones_like(t['labels'], dtype=torch.float32)} for t in preds]

        super().update(preds, target)

    def compute(self):
        '''
        Compute metric
        Returns:
            metric: metric value mAP
        '''
        metric = super().compute()
      
        return metric['map']

    
def get_metrics(metric_names: list[str], num_classes: int, prefix: str) -> Dict[str, Union[Accuracy, CustomMAP, Dice]]:
    """
    Set up metrics for evaluation
    Args:
        metric_names: list of metric names
        num_classes: number of classes for relevant metrics
    Returns:
        metrics: dictionary of metric instances
    """
    metric_mapping = {
        'accuracy': lambda: Accuracy(task='multiclass', num_classes=num_classes),
        'f1': lambda: F1Score(task='multiclass', num_classes=num_classes, average='macro'),
        'map': CustomMAP,
        'dice': lambda: Dice(num_classes=num_classes),
    }

    metrics = {}

    for metric_name in metric_names:
        metric_creator = metric_mapping.get(metric_name)

        if metric_creator is not None:
            metrics[metric_name] = metric_creator()
        else:
            raise NotImplementedError(f"Metric '{metric_name}' is not implemented.")

    return MetricCollection(metrics, prefix=prefix)