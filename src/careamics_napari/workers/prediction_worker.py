"""A thread worker function running CAREamics prediction."""

import traceback
from collections.abc import Generator
from queue import Queue
from threading import Thread

from careamics import CAREamist
from numpy.typing import NDArray
from superqt.utils import thread_worker

from careamics_napari.careamics_utils import BaseConfig
from careamics_napari.signals import (
    PredictionState,
    PredictionUpdate,
    PredictionUpdateType,
)


@thread_worker
def predict_worker(
    careamist: CAREamist,
    pred_data: NDArray | str,
    configuration: BaseConfig,
    update_queue: Queue,
) -> Generator[PredictionUpdate, None, None]:
    """Model prediction worker.

    Parameters
    ----------
    careamist : CAREamist
        CAREamist instance.
    pred_data : NDArray | str
        Prediction data source.
    configuration : BaseConfig
        careamics configuration.
    update_queue : Queue
        Queue used to send updates to the UI.

    Yields
    ------
    Generator[PredictionUpdate, None, None]
        Updates.
    """
    # start training thread
    training = Thread(
        target=_predict,
        args=(
            careamist,
            pred_data,
            configuration,
            update_queue,
        ),
    )
    training.start()

    # look for updates
    while True:
        update: PredictionUpdate = update_queue.get(block=True)

        yield update

        if (
            update.type == PredictionUpdateType.STATE
            or update.type == PredictionUpdateType.EXCEPTION
        ):
            break


def _push_exception(queue: Queue, e: Exception) -> None:
    """Push an exception to the queue.

    Parameters
    ----------
    queue : Queue
        Queue.
    e : Exception
        Exception.
    """
    try:
        raise e
    except Exception:
        traceback.print_exc()

    queue.put(PredictionUpdate(PredictionUpdateType.EXCEPTION, e))


def _predict(
    careamist: CAREamist,
    pred_data: NDArray | str,
    configuration: BaseConfig,
    update_queue: Queue,
) -> None:
    """Run the prediction.

    Parameters
    ----------
    careamist : CAREamist
        CAREamist instance.
    data_source : NDArray | str
        Prediction data source.
    configuration : BaseConfig
        careamics configuration.
    update_queue : Queue
        Queue used to send updates to the UI.
    """
    data_type = "tiff" if isinstance(pred_data, str) else "array"
    tile_overlap = [configuration.tile_overlap_xy, configuration.tile_overlap_xy]
    if configuration.is_3D:
        tile_overlap.insert(0, configuration.tile_overlap_z)
    # predict with careamist
    try:
        result = careamist.predict(  # type: ignore
            pred_data,  # type: ignore
            data_type=data_type,  # type: ignore
            tile_size=configuration.tile_size,
            tile_overlap=tuple(tile_overlap),
            batch_size=configuration.pred_batch_size,
        )

        if result is not None and len(result) > 0:
            update_queue.put(PredictionUpdate(PredictionUpdateType.SAMPLE, result))

    except Exception as e:
        traceback.print_exc()

        update_queue.put(PredictionUpdate(PredictionUpdateType.EXCEPTION, e))
        return

    # signify end of prediction
    update_queue.put(PredictionUpdate(PredictionUpdateType.STATE, PredictionState.DONE))
