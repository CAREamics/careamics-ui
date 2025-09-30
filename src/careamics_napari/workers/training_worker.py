"""A thread worker function running CAREamics training."""

import traceback
from collections.abc import Generator
from queue import Queue
from threading import Thread

import napari.utils.notifications as ntf
import numpy as np
from careamics import CAREamist
from superqt.utils import thread_worker

from careamics_napari.careamics_utils import BaseConfig, UpdaterCallBack
from careamics_napari.signals import (
    TrainingState,
    TrainUpdate,
    TrainUpdateType,
)


# TODO register CAREamist to continue training and predict
# TODO how to load pre-trained?
@thread_worker
def train_worker(
    configuration: BaseConfig,
    data_sources: dict[str, list],
    training_queue: Queue,
    predict_queue: Queue,
    careamist: CAREamist | None = None,
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
    """
    train_data = data_sources["train"][0]
    val_data = data_sources["val"][0]
    train_target = None
    val_target = None
    # check if target data is available
    if len(data_sources["train"]) > 1:
        train_target = data_sources["train"][1]
    if len(data_sources["val"]) > 1:
        val_target = data_sources["val"][1]

    # if the train and validation data are the same
    if np.array_equal(train_data, val_data):
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
            careamist = CAREamist(
                configuration,
                work_dir=configuration.work_dir,
                callbacks=[UpdaterCallBack(training_queue, predict_queue)],
            )
        else:
            # only update the number of epochs
            if configuration.training_config.lightning_trainer_config:
                configuration.training_config.lightning_trainer_config["max_epochs"] = (
                    configuration.training_config.lightning_trainer_config["max_epochs"]
                )
            if val_data is None:
                ntf.show_error(
                    "Continuing training is currently not supported without explicitely "
                    "passing validation. The reason is that otherwise, the data used for "
                    "validation will be different and there will be data leakage in the "
                    "training set."
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
                train_source=train_data,
                val_source=val_data,
                train_target=train_target,
                val_target=val_target,
                val_minimum_split=configuration.val_minimum_split,
                val_percentage=configuration.val_percentage,
            )
        except Exception as e:
            traceback.print_exc()
            training_queue.put(TrainUpdate(TrainUpdateType.EXCEPTION, e))

    training_queue.put(TrainUpdate(TrainUpdateType.STATE, TrainingState.DONE))
