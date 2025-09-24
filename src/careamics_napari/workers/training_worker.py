"""A thread worker function running CAREamics training."""

import traceback
from collections.abc import Generator
from queue import Queue
from threading import Thread
from typing import Optional

import napari.utils.notifications as ntf
from careamics import CAREamist
from careamics.config.support import SupportedAlgorithm
from superqt.utils import thread_worker

from careamics_napari.careamics_utils import UpdaterCallBack
from careamics_napari.careamics_utils.configuration import create_configuration
from careamics_napari.signals import (
    PredictionStatus,
    TrainingSignal,
    TrainingState,
    TrainUpdate,
    TrainUpdateType,
)
from careamics_napari.utils.prediction_callback import StopPredictionCallback


# TODO register CAREamist to continue training and predict
# TODO how to load pre-trained?
# TODO pass careamist here if it already exists?
@thread_worker
def train_worker(
    train_config_signal: TrainingSignal,
    training_queue: Queue,
    predict_queue: Queue,
    careamist: Optional[CAREamist] = None,
    pred_status: Optional[PredictionStatus] = None,
) -> Generator[TrainUpdate, None, None]:
    """Model training worker.

    Parameters
    ----------
    train_config_signal : TrainingSignal
        Training signal.
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
            train_config_signal,
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
    config_signal: TrainingSignal,
    training_queue: Queue,
    predict_queue: Queue,
    careamist: Optional[CAREamist] = None,
    pred_status: Optional[PredictionStatus] = None,
) -> None:
    """Run the training.

    Parameters
    ----------
    config_signal : TrainingSignal
        Training signal.
    training_queue : Queue
        Training update queue.
    predict_queue : Queue
        Prediction update queue.
    careamist : CAREamist or None, default=None
        CAREamist instance.
    pred_status : PredictionStatus or None, default=None
        Prediction status for stop callback.
    """
    # get configuration and queue
    try:
        # create_configuration can raise an exception
        config = create_configuration(config_signal)

        # Create CAREamist
        if careamist is None:
            callbacks = [UpdaterCallBack(training_queue, predict_queue)]
            if pred_status is not None:
                callbacks.append(StopPredictionCallback(pred_status))
            careamist = CAREamist(config, callbacks=callbacks)

        else:
            # only update the number of epochs
            careamist.cfg.training_config.num_epochs = config.training_config.num_epochs

            if config_signal.layer_val == "" and config_signal.path_val == "":
                ntf.show_error(
                    "Continuing training is currently not supported without explicitely "
                    "passing validation. The reason is that otherwise, the data used for "
                    "validation will be different and there will be data leakage in the "
                    "training set."
                )
    except Exception as e:
        traceback.print_exc()

        training_queue.put(TrainUpdate(TrainUpdateType.EXCEPTION, e))

    # Register CAREamist
    training_queue.put(TrainUpdate(TrainUpdateType.CAREAMIST, careamist))

    # Format data
    train_data_target = None
    val_data_target = None

    if config_signal.load_from_disk:

        if config_signal.path_train == "":
            _push_exception(training_queue, ValueError("Training data path is empty."))
            return

        train_data = config_signal.path_train
        val_data = config_signal.path_val if config_signal.path_val != "" else None

        if train_data == val_data:
            val_data = None

        if config_signal.algorithm != SupportedAlgorithm.N2V:
            if config_signal.path_train_target == "":
                _push_exception(
                    training_queue, ValueError("Training target data path is empty.")
                )
                return

            train_data_target = config_signal.path_train_target

            if val_data is not None:
                val_data_target = (
                    config_signal.path_val_target
                    if config_signal.path_val_target != ""
                    else None
                )

    else:
        if config_signal.layer_train is None:
            _push_exception(
                training_queue, ValueError("Training layer has not been selected.")
            )
            return

        elif config_signal.layer_train.data is None:
            _push_exception(
                training_queue,
                ValueError(
                    f"Training layer {config_signal.layer_train.name} is empty."
                ),
            )
            return
        else:
            train_data = config_signal.layer_train.data

        val_data = (
            config_signal.layer_val.data
            if config_signal.layer_val is not None
            and config_signal.layer_val.data is not None
            else None
        )

        if (
            config_signal.layer_train is not None
            and config_signal.layer_val is not None
            and (config_signal.layer_train.name == config_signal.layer_val.name)
        ):
            val_data = None

        if config_signal.algorithm != SupportedAlgorithm.N2V:

            if config_signal.layer_train_target is None:
                _push_exception(
                    training_queue,
                    ValueError("Training target layer has not been selected."),
                )
                return
            elif config_signal.layer_train_target.data is None:
                _push_exception(
                    training_queue,
                    ValueError(
                        f"Training target layer {config_signal.layer_train_target.name}"
                        f" is empty."
                    ),
                )
                return
            else:
                train_data_target = config_signal.layer_train_target.data

            if val_data is not None:
                val_data_target = (
                    config_signal.layer_val_target.data
                    if config_signal.layer_val_target is not None
                    and config_signal.layer_val_target.data is not None
                    else None
                )
            else:
                val_data_target = None

    # TODO add val percentage and val minimum
    # Train CAREamist
    try:
        careamist.train(
            train_source=train_data,
            val_source=val_data,
            train_target=train_data_target,
            val_target=val_data_target,
            val_minimum_split=config_signal.val_minimum_split,
            val_percentage=config_signal.val_percentage,
        )

        # # TODO can we use this to monkey patch the training process?
        # update_queue.put(Update(UpdateType.MAX_EPOCH, 10_000 // 10))
        # update_queue.put(Update(UpdateType.MAX_BATCH, 10_000))
        # for i in range(10_000):

        #     # if stopper.stop:
        #     #     update_queue.put(Update(UpdateType.STATE, TrainingState.STOPPED))
        #     #     break

        #     if i % 10 == 0:
        #         update_queue.put(Update(UpdateType.EPOCH, i // 10))
        #         print(i)

        #     update_queue.put(Update(UpdateType.BATCH, i))

        #     time.sleep(0.2)

    except Exception as e:
        traceback.print_exc()

        training_queue.put(TrainUpdate(TrainUpdateType.EXCEPTION, e))

    training_queue.put(TrainUpdate(TrainUpdateType.STATE, TrainingState.DONE))
