"""A thread worker function running CAREamics training."""

import traceback
from collections.abc import Generator
from queue import Queue
from threading import Thread

import napari.utils.notifications as ntf
import numpy as np
from careamics import CAREamist
from careamics.lightning import StopPredictionCallback
from numpy.typing import NDArray
from superqt.utils import thread_worker

from careamics_napari.careamics_utils import (
    BaseConfig,
    DiskWriterCallback,
    UpdaterCallBack,
)
from careamics_napari.signals import (
    PredictionState,
    PredictionStatus,
    TrainingState,
    TrainUpdate,
    TrainUpdateType,
)


def is_val_the_same_as_train(
    train_data: NDArray | str,
    val_data: NDArray | str | None,
) -> bool:
    """Check if validation data is the same as training data.

    Parameters
    ----------
    train_data : NDArray | str
        Training data.
    val_data : NDArray | str | None
        Validation data.

    Returns
    -------
    bool
        True if validation data is the same as training data, False otherwise.
    """
    if val_data is None:
        return False

    if isinstance(train_data, str):
        return train_data == val_data

    if isinstance(val_data, np.ndarray):
        return np.array_equal(train_data, val_data)

    return False


@thread_worker
def train_worker(
    configuration: BaseConfig,
    data_sources: dict[str, list],
    training_queue: Queue,
    predict_queue: Queue,
    careamist: CAREamist | None = None,
    pred_status: PredictionStatus | None = None,
) -> Generator[TrainUpdate, None, None]:
    """Model training worker.

    Parameters
    ----------
    configuration : BaseConfig
        careamics configuration.
    data_sources : dict[str, list]
        Train and validation data sources.
    training_queue : Queue
        Training update queue.
    predict_queue : Queue
        Prediction update queue.
    careamist : CAREamist or None, default=None
        CAREamist instance.
    pred_status : PredictionStatus or None, default=None
        Prediction status for stop callback.

    Yields
    ------
    Generator[TrainUpdate, None, None]
        Updates.
    """
    # start training thread
    training = Thread(
        target=_train,
        args=(
            configuration,
            data_sources,
            training_queue,
            predict_queue,
            careamist,
            pred_status,
        ),
    )
    training.start()

    # look for updates
    while True:
        update: TrainUpdate = training_queue.get(block=True)

        yield update

        if (
            update.type == TrainUpdateType.STATE and update.value == TrainingState.DONE
        ) or (update.type == TrainUpdateType.EXCEPTION):
            break

    # wait for the other thread to finish
    training.join()


def _push_exception(queue: Queue, e: Exception) -> None:
    """Push an exception to the queue.

    Parameters
    ----------
    queue : Queue
        Queue.
    e : Exception
        Exception.
    """
    queue.put(TrainUpdate(TrainUpdateType.EXCEPTION, e))


def _train(
    configuration: BaseConfig,
    data_sources: dict[str, list],
    training_queue: Queue,
    predict_queue: Queue,
    careamist: CAREamist | None = None,
    pred_status: PredictionStatus | None = None,
) -> None:
    """Run the training.

    Parameters
    ----------
    configuration : BaseConfig
        careamics configuration.
    data_sources : dict[str, list]
        Train and validation data sources.
    training_queue : Queue
        Training update queue.
    predict_queue : Queue
        Prediction update queue.
    careamist : CAREamist or None, default=None
        CAREamist instance.
    pred_status : PredictionStatus or None, default=None
        Prediction status for stop callback.
    """
    train_data: NDArray | str = data_sources["train"][0]
    val_data: NDArray | str | None = data_sources["val"][0]

    # check if val data is set
    # in case data is coming from files from disk
    if isinstance(val_data, str) and len(val_data) == 0:
        val_data = None

    # check if target data is available
    train_target = None
    val_target = None
    if len(data_sources["train"]) > 1:
        train_target = data_sources["train"][1]
    if len(data_sources["val"]) > 1:
        val_target = data_sources["val"][1]

    # if the train and validation data are the same
    if is_val_the_same_as_train(train_data, val_data):
        val_data = None
        val_target = None

    # check if algorithm needs GT and it's available
    if configuration.needs_gt and train_target is None:
        _push_exception(
            training_queue,
            ValueError("Training target (GT) is required but wasn't provided."),
        )

    try:
        # create / update CAREamist
        if careamist is None:
            callbacks: list = [
                UpdaterCallBack(training_queue, predict_queue),
                DiskWriterCallback(training_queue, predict_queue),
            ]
            if pred_status is not None:
                callbacks.append(
                    StopPredictionCallback(
                        lambda: pred_status.state == PredictionState.STOPPED
                    )
                )

            careamist = CAREamist(
                configuration, callbacks=callbacks, work_dir=configuration.work_dir
            )
        else:
            if val_data is None:
                ntf.show_error(
                    "Continuing training is currently not supported without explicitely "
                    "passing validation. The reason is that otherwise, the data used for "
                    "validation will be different and there will be data leakage in the "
                    "training set."
                )
            # update the number of epochs and number of steps
            if configuration.training_config.trainer_params:
                careamist.config.training_config.trainer_params["max_epochs"] = (
                    configuration.training_config.trainer_params["max_epochs"]
                )
                careamist.config.training_config.trainer_params["limit_train_batches"] = (
                    configuration.training_config.trainer_params["limit_train_batches"]
                )
    except Exception as e:
        traceback.print_exc()
        training_queue.put(TrainUpdate(TrainUpdateType.EXCEPTION, e))

    # register CAREamist
    training_queue.put(TrainUpdate(TrainUpdateType.CAREAMIST, careamist))

    # train CAREamist
    if careamist is not None:
        try:
            careamist.train(
                train_data=train_data,
                train_data_target=train_target,
                val_data=val_data,
                val_data_target=val_target,
            )
        except Exception as e:
            traceback.print_exc()
            training_queue.put(TrainUpdate(TrainUpdateType.EXCEPTION, e))

    training_queue.put(TrainUpdate(TrainUpdateType.STATE, TrainingState.DONE))
