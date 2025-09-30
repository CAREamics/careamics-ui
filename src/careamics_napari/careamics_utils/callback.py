"""PyTorch Lightning callback used to update GUI with progress."""

from queue import Queue
from typing import Any

import numpy as np
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from typing_extensions import Self

from careamics_napari.signals import (
    PredictionUpdate,
    PredictionUpdateType,
    TrainUpdate,
    TrainUpdateType,
)


class UpdaterCallBack(Callback):
    """PyTorch Lightning callback for updating training and prediction UI states.

    Parameters
    ----------
    training_queue : Queue
        Training queue used to pass updates between threads.
    prediction_queue : Queue
        Prediction queue used to pass updates between threads.

    Attributes
    ----------
    training_queue : Queue
        Training queue used to pass updates between threads.
    prediction_queue : Queue
        Prediction queue used to pass updates between threads.
    """

    def __init__(self: Self, training_queue: Queue, prediction_queue: Queue) -> None:
        """Initialize the callback.

        Parameters
        ----------
        training_queue : Queue
            Training queue used to pass updates between threads.
        prediction_queue : Queue
            Prediction queue used to pass updates between threads.
        """
        self.training_queue = training_queue
        self.prediction_queue = prediction_queue

    def get_train_queue(self) -> Queue:
        """Return the training queue.

        Returns
        -------
        Queue
            Training queue.
        """
        return self.training_queue

    def get_predict_queue(self) -> Queue:
        """Return the prediction queue.

        Returns
        -------
        Queue
            Prediction queue.
        """
        return self.prediction_queue

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Method called at the beginning of the training.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning trainer.
        pl_module : LightningModule
            PyTorch Lightning module.
        """
        # compute the number of batches
        len_dataloader = len(trainer.train_dataloader)  # type: ignore

        self.training_queue.put(
            TrainUpdate(
                TrainUpdateType.MAX_BATCH,
                int(len_dataloader / trainer.accumulate_grad_batches),
            )
        )

        # register number of epochs
        self.training_queue.put(
            TrainUpdate(TrainUpdateType.MAX_EPOCH, trainer.max_epochs)
        )

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Method called at the beginning of each epoch.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning trainer.
        pl_module : LightningModule
            PyTorch Lightning module.
        """
        self.training_queue.put(TrainUpdate(TrainUpdateType.EPOCH, trainer.current_epoch))

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Method called at the end of each epoch.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning trainer.
        pl_module : LightningModule
            PyTorch Lightning module.
        """
        metrics = trainer.progress_bar_metrics

        if "train_loss_epoch" in metrics:
            self.training_queue.put(
                TrainUpdate(TrainUpdateType.LOSS, metrics["train_loss_epoch"])
            )

        if "val_loss" in metrics:
            self.training_queue.put(
                TrainUpdate(TrainUpdateType.VAL_LOSS, metrics["val_loss"])
            )

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        """Method called at the beginning of each batch.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning trainer.
        pl_module : LightningModule
            PyTorch Lightning module.
        batch : Any
            Batch.
        batch_idx : int
            Index of the batch.
        """
        self.training_queue.put(TrainUpdate(TrainUpdateType.BATCH, batch_idx))

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Method called at the beginning of the prediction.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning trainer.
        pl_module : LightningModule
            PyTorch Lightning module.
        """
        # lightning returns a number of batches per dataloader
        # if data is loading from disk, the IterableDataset length is not defined.
        n_batches = trainer.num_predict_batches[0]
        if n_batches == np.inf:
            n_batches = "?"
        else:
            n_batches = int(n_batches)

        self.prediction_queue.put(
            PredictionUpdate(
                PredictionUpdateType.MAX_SAMPLES,
                n_batches,
            )
        )

    def on_predict_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Method called at the beginning of each prediction batch.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning trainer.
        pl_module : LightningModule
            PyTorch Lightning module.
        batch : Any
            Batch.
        batch_idx : int
            Index of the batch.
        dataloader_idx : int, default=0
            Index of the dataloader.
        """
        self.prediction_queue.put(
            PredictionUpdate(PredictionUpdateType.SAMPLE_IDX, batch_idx)
        )
