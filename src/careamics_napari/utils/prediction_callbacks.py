"""Custom PyTorch Lightning callbacks for prediction control."""

from threading import Event
from typing import Any

import pytorch_lightning as pl
from typing_extensions import Self


class PredictionStoppedException(Exception):
    """Exception raised when prediction is stopped by user."""
    pass


class StopPredictionCallback(pl.Callback):
    """PyTorch Lightning callback to stop prediction when signaled.
    
    This callback monitors a threading.Event and stops the trainer
    when the event is set, allowing for graceful interruption of
    prediction processes.
    
    Parameters
    ----------
    stop_event : threading.Event
        Event that when set, signals the prediction to stop.
    """
    
    def __init__(self: Self, stop_event: Event) -> None:
        """Initialize the callback.
        
        Parameters
        ----------
        stop_event : threading.Event
            Event that when set, signals the prediction to stop.
        """
        super().__init__()
        self.stop_event = stop_event
    
    def on_predict_batch_start(
        self: Self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        batch: Any, 
        batch_idx: int, 
        dataloader_idx: int = 0
    ) -> None:
        """Check for stop signal at the start of each prediction batch.
        
        Parameters
        ----------
        trainer : pl.Trainer
            The PyTorch Lightning trainer.
        pl_module : pl.LightningModule  
            The Lightning module being used.
        batch : Any
            The current batch of data.
        batch_idx : int
            Index of the current batch.
        dataloader_idx : int, optional
            Index of the current dataloader, by default 0.
        """
        if self.stop_event.is_set():
            print("Stop signal received, stopping prediction...")
            trainer.should_stop = True
            # For prediction, we need to raise an exception to actually stop
            raise PredictionStoppedException("Prediction stopped by user")
