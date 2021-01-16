from typing import Tuple

from overrides import overrides

import torch
import torch.distributed as dist

from allennlp.common.util import is_distributed
from allennlp.training.metrics.metric import Metric
from allennlp_models.rc.tools import squad


@Metric.register("squad_em_and_f1")
class Squad2EmAndF1(Metric):
    """
    This :class:`Metric` takes the best span string computed by a model, along with the answer
    strings labeled in the data, and computed exact match and F1 score using the official SQuAD2
    evaluation script.
    """

    supports_distributed = True

    def __init__(self, is_main=False) -> None:
        self.is_main = is_main

        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0

    @overrides
    def __call__(self, best_span_string, gold_answers):
        exact_match = float(
            max(squad.compute_exact(gold_answer, best_span_string) for gold_answer in gold_answers)
        )
        f1_score = float(
            max(squad.compute_f1(gold_answer, best_span_string) for gold_answer in gold_answers)
        )

        self._total_em += exact_match
        self._total_f1 += f1_score
        self._count += 1

        return exact_match, f1_score

    @overrides
    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        """
        Returns
        -------
        Average exact match and F1 score (in that order) as computed by the official SQuAD script
        over all inputs.
        """
        if is_distributed():
            device = "cuda" if dist.get_backend() == "nccl" else "cpu"
        else:
            device = "cpu"

        _total_em = torch.tensor(self._total_em, device=device)
        _total_f1 = torch.tensor(self._total_f1, device=device)
        _count = torch.tensor(self._count, device=device)
        if is_distributed():
            dist.all_reduce(_total_em, op=dist.ReduceOp.SUM)
            dist.all_reduce(_total_f1, op=dist.ReduceOp.SUM)
            dist.all_reduce(_count, op=dist.ReduceOp.SUM)

        if reset:
            self.reset()

        exact_match = _total_em.item() / _count.item() if _count > 0 else 0
        f1_score = _total_f1.item() / _count.item() if _count > 0 else 0
        count = _count.item()

        return {"em": exact_match, "f1": f1_score, "count": count}

    @overrides
    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0

    def __str__(self):
        return f"Squad2EmAndF1(em={self._total_em}, f1={self._total_f1}, count={self._count})"
