from collections import deque
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Optional

import numpy as np

from careamics import CAREamist
from careamics.config.support import SupportedAlgorithm
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QStackedWidget, QVBoxLayout, QWidget
from typing_extensions import Self

from careamics_napari.signals import (
    PredictionSignal,
    PredictionState,
    PredictionStatus,
    PredictionUpdate,
    PredictionUpdateType,
    SavingSignal,
    SavingState,
    SavingStatus,
    SavingUpdate,
    SavingUpdateType,
    TrainingState,
    TrainingStatus,
    TrainUpdate,
    TrainUpdateType,
    N2VTrainingSignal
)
from careamics_napari.workers import predict_worker, save_worker, train_worker
from careamics_napari.utils.axes_utils import reshape_prediction
from careamics_napari.base_plugin import BasePlugin


if TYPE_CHECKING:
    import napari

# at run time
try:
    import napari
    import napari.utils.notifications as ntf

except ImportError:
    _has_napari = False
else:
    _has_napari = True


class N2VPlugin(BasePlugin):
    """CAREamics N2V plugin.

    Parameters
    ----------
    napari_viewer : napari.Viewer or None, default=None
        Napari viewer.
    """

    def __init__(
        self: Self,
        napari_viewer: Optional[napari.Viewer] = None,
    ) -> None:
        """Initialize the plugin.

        Parameters
        ----------
        napari_viewer : napari.Viewer or None, default=None
            Napari viewer.
        """
        super().__init__()
        self.viewer = napari_viewer
        self.careamist: Optional[CAREamist] = None
        self.train_config_signal = N2VTrainingSignal()  # type: ignore

        self.add_careamics_banner(
            "CAREamics UI for training N2V denoising model."
        )
        self.add_train_input_ui(use_target=False)
        self.add_config_ui()
        self.add_train_button_ui()
        self.add_prediction_ui()
        self.add_model_saving_ui()


if __name__ == "__main__":
    import faulthandler
    import napari

    log_file_fd = open("fault_log.txt", "a")
    faulthandler.enable(log_file_fd)
    # create a Viewer
    viewer = napari.Viewer()
    # add n2v plugin
    viewer.window.add_dock_widget(N2VPlugin(viewer))
    # start UI
    napari.run()
