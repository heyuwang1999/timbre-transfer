"""Dataset preprocessing and fine-tuning entry points.

IMPORTANT: a RAVE ``.ts`` export is inference-only and cannot be fine-tuned.
Genuine RAVE training/fine-tuning uses the ``acids-rave`` package (the ``rave``
CLI) operating on a Lightning ``.ckpt``. See :mod:`timbre_transfer.training.finetune`.
"""

from .finetune import finetune_illustrative, rave_train
from .preprocess import preprocess_dataset, rave_preprocess

__all__ = [
    "preprocess_dataset",
    "rave_preprocess",
    "finetune_illustrative",
    "rave_train",
]
