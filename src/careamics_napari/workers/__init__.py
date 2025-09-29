"""Callable used to run the workers in a new thread."""

__all__ = ["predict_worker", "train_worker"]

from .prediction_worker import predict_worker
from .training_worker import train_worker
