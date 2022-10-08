from typing import Optional, List
from functools import partial
import torch
from torchmetrics import Metric
import torchseg.metrics.functional as MF


class MyMetric(Metric):
    full_state_update = True

    def __init__(self, mode: str,
                 ignore_index: Optional[int] = None,
                 threshold: Optional[float] = None,
                 num_classes: Optional[int] = None,
                 reduction: Optional[str] = None,
                 class_weights: Optional[List[float]] = None):
        super().__init__()
        self.mode = mode
        self.ignore_index = ignore_index
        self.threshold = threshold
        self.num_classes = num_classes
        self.get_stats = partial(MF.get_stats,
                                 mode=self.mode,
                                 ignore_index=self.ignore_index,
                                 threshold=self.threshold,
                                 num_classes=self.num_classes)

        self.reduction = reduction
        self.class_weights = class_weights

        self.add_state("tp", default=torch.tensor(0, dtype=torch.long))
        self.add_state("fp", default=torch.tensor(0, dtype=torch.long))
        self.add_state("tn", default=torch.tensor(0, dtype=torch.long))
        self.add_state("fn", default=torch.tensor(0, dtype=torch.long))

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        tp, fp, fn, tn = self.get_stats(pred, target)

        # Reshape needed because at first self.tp is singleton
        if self.tp.ndim == 0:
            # TODO: Understand how not to have to push to right device here!
            self.tp = self.tp.repeat(1, pred.shape[1]).to(tp.device)
            self.fp = self.fp.repeat(1, pred.shape[1]).to(tp.device)
            self.tn = self.tn.repeat(1, pred.shape[1]).to(tp.device)
            self.fn = self.fn.repeat(1, pred.shape[1]).to(tp.device)

        self.tp = torch.cat([self.tp, tp], dim=0)
        self.fp = torch.cat([self.fp, fp], dim=0)
        self.tn = torch.cat([self.tn, tn], dim=0)
        self.fn = torch.cat([self.fn, fn], dim=0)


class FBetaScore(MyMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        return MF.fbeta_score(self.tp, self.fp, self.fn, self.tn,
                              reduction=self.reduction,
                              class_weights=self.class_weights)


class F1Score(MyMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        return MF.f1_score(self.tp, self.fp, self.fn, self.tn,
                           reduction=self.reduction,
                           class_weights=self.class_weights)


class IoUScore(MyMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        return MF.iou_score(self.tp, self.fp, self.fn, self.tn,
                            reduction=self.reduction,
                            class_weights=self.class_weights)


class Accuracy(MyMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        return MF.accuracy(self.tp, self.fp, self.fn, self.tn,
                           reduction=self.reduction,
                           class_weights=self.class_weights)


class Precision(MyMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        return MF.precision(self.tp, self.fp, self.fn, self.tn,
                            reduction=self.reduction,
                            class_weights=self.class_weights)


class Recall(MyMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        return MF.recall(self.tp, self.fp, self.fn, self.tn,
                         reduction=self.reduction,
                         class_weights=self.class_weights)


class Sensitivity(MyMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        return MF.sensitivity(self.tp, self.fp, self.fn, self.tn,
                              reduction=self.reduction,
                              class_weights=self.class_weights)


class Specificity(MyMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        return MF.specificity(self.tp, self.fp, self.fn, self.tn,
                              reduction=self.reduction,
                              class_weights=self.class_weights)


class BalancedAccuracy(MyMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        return MF.balanced_accuracy(self.tp, self.fp, self.fn, self.tn,
                                    reduction=self.reduction,
                                    class_weights=self.class_weights)
