"""A thread worker function running CAREamics prediction."""

import traceback
from collections.abc import Generator

from careamics import CAREamist
from superqt.utils import thread_worker

from careamics_napari.signals import (
    ExportType,
    SavingSignal,
    SavingState,
    SavingUpdate,
    SavingUpdateType,
    TrainingSignal,
)


@thread_worker
def save_worker(
    careamist: CAREamist,
    training_signal: TrainingSignal,
    config_signal: SavingSignal,
) -> Generator[SavingUpdate, None, None]:
    """Model saving worker.

    Parameters
    ----------
    careamist : CAREamist
        CAREamist instance.
    training_signal : TrainingSignal
        Training signal.
    config_signal : SavingSignal
        Saving signal.

    Yields
    ------
    Generator[SavingUpdate, None, None]
        Updates.

    Raises
    ------
    NotImplementedError
        Export to BMZ not implemented yet.
    """
    dims = "3D" if training_signal.is_3d else "2D"
    name = f"{training_signal.algorithm}_{dims}_{training_signal.experiment_name}"

    # save model
    try:
        if config_signal.export_type == ExportType.BMZ:
            raise NotImplementedError("Export to BMZ not implemented yet (but soon).")

        else:
            name = name + ".ckpt"
            # TODO: should we reexport the model every time?
            careamist.trainer.save_checkpoint(
                config_signal.path_model / name,
            )

    except Exception as e:
        traceback.print_exc()

        yield SavingUpdate(SavingUpdateType.EXCEPTION, e)

    yield SavingUpdate(SavingUpdateType.STATE, SavingState.DONE)
