"""A thread worker function running CAREamics prediction."""

import traceback
from collections.abc import Generator
from pathlib import Path
from queue import Queue
from threading import Thread

# from careamics.lightning import PredictionStoppedException
from careamics.careamist_v2 import CAREamistV2
from numpy.typing import NDArray
from superqt.utils import thread_worker

from careamics_napari.careamics_utils import BaseConfig, PredictionStoppedException
from careamics_napari.signals import (
    PredictionState,
    PredictionUpdate,
    PredictionUpdateType,
)


@thread_worker
def predict_worker(
    careamist: CAREamistV2,
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
    # start prediction thread
    prediction = Thread(
        target=_predict,
        args=(
            careamist,
            pred_data,
            configuration,
            update_queue,
        ),
    )
    prediction.start()

    # look for updates
    while True:
        update: PredictionUpdate = update_queue.get(block=True)

        yield update

        if (
            update.type == PredictionUpdateType.STATE
            or update.type == PredictionUpdateType.EXCEPTION
        ):
            break


def _predict(
    careamist: CAREamistV2,
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
    data_type = "array"
    write_type = "tiff"
    in_memory = True
    if isinstance(pred_data, str | Path):
        if str(pred_data).endswith("zarr"):
            data_type = "zarr"
            write_type = "zarr"
            in_memory = False
        else:
            data_type = "tiff"
        # prediction dir if writing to disk is set
        pred_dir = Path(pred_data).parent / "predictions"
    else:
        pred_dir = configuration.work_dir / "predictions"

    tile_overlap = [configuration.tile_overlap_xy, configuration.tile_overlap_xy]
    if configuration.is_3D:
        tile_overlap.insert(0, configuration.tile_overlap_z)

    # predict with careamist
    try:
        if configuration.write_to_disk:
            result = careamist.predict_to_disk(
                pred_data,
                prediction_dir=pred_dir,
                data_type=data_type,
                write_type=write_type,
                tile_size=configuration.tile_size,
                tile_overlap=tuple(tile_overlap),
                batch_size=configuration.pred_batch_size,
                in_memory=in_memory,
            )
            print(f"Predictions are written to {pred_dir}")

        else:
            result, _ = careamist.predict(
                pred_data,
                data_type=data_type,
                tile_size=configuration.tile_size,
                tile_overlap=tuple(tile_overlap),
                batch_size=configuration.pred_batch_size,
            )

        if result is not None and len(result) > 0:
            update_queue.put(PredictionUpdate(PredictionUpdateType.SAMPLE, result))

    except PredictionStoppedException:
        # handle user-requested stop
        update_queue.put(
            PredictionUpdate(PredictionUpdateType.STATE, PredictionState.STOPPED)
        )
        return

    except Exception as e:
        traceback.print_exc()

        update_queue.put(PredictionUpdate(PredictionUpdateType.EXCEPTION, e))
        return

    # signify end of prediction
    update_queue.put(PredictionUpdate(PredictionUpdateType.STATE, PredictionState.DONE))
