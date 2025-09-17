"""CAREamics prediction Qt widget."""

from collections import deque
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QWidget,
    QLabel, QPushButton, QLineEdit, QFileDialog
)
from typing_extensions import Self

from careamics import CAREamist
from careamics_napari.signals import (
    PredictionSignal,
    PredictionState,
    PredictionStatus,
    PredictionUpdate,
    PredictionUpdateType,
    TrainingState,
    TrainingStatus,
)
from careamics_napari.widgets import (
    CAREamicsBanner,
    PredictionWidget,
    ScrollWidgetWrapper,
    create_gpu_label,
)
from careamics_napari.careamics_utils import UpdaterCallBack
from careamics_napari.workers import predict_worker
from careamics_napari.utils.axes_utils import reshape_prediction

import numpy as np

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


class PredictionPluginWrapper(ScrollWidgetWrapper):
    """Training plugin within a scrolling wrapper.

    Parameters
    ----------
    napari_viewer : napari.Viewer or None, default=None
        Napari viewer.
    """

    def __init__(self: Self, napari_viewer: Optional[napari.Viewer] = None) -> None:
        """Initialize the plugin.

        Parameters
        ----------
        napari_viewer : napari.Viewer or None, default=None
            Napari viewer.
        """
        super().__init__(PredictionPlugin(napari_viewer))


class PredictionPlugin(QWidget):
    """CAREamics prediction plugin.

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

        # create statuses, used to keep track of the threads statuses
        # TODO: prediction widget should not be dependent on the training status
        self.train_status = TrainingStatus()  # type: ignore
        self.pred_status = PredictionStatus()  # type: ignore

        # create signals, used to hold the various parameters modified by the UI
        self.pred_config_signal = PredictionSignal()

        # create queues, used to communicate between the threads and the UI
        # TODO: we shouldn't need to have a training queue here
        # right now, UpdateCallBack init requires it.
        self._training_queue: Queue = Queue(10)
        self._prediction_queue: Queue = Queue(10)

        self._init_ui()

    def _init_ui(self) -> None:
        """Assemble the widgets."""
        # layout
        self.setLayout(QVBoxLayout())
        self.setMinimumWidth(200)

        # add banner
        self.layout().addWidget(
            CAREamicsBanner(
                title="CAREamics",
                short_desc=("CAREamics UI for training denoising models."),
            )
        )
        # algorithm
        self.algo_label = QLabel("**Algorithm**: *model is not loaded*")
        self.algo_label.setTextFormat(Qt.MarkdownText)
        self.algo_label.setEnabled(False)
        # gpu button
        gpu_button = create_gpu_label()
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 5)  # bottom margin
        hbox.addWidget(self.algo_label)
        hbox.addWidget(gpu_button, alignment=Qt.AlignmentFlag.AlignRight)
        self.layout().addLayout(hbox)

        # load model ui
        load_button = QPushButton("Load Model...", self)
        load_button.clicked.connect(self._select_model_checkpoint)
        self.model_textbox = QLineEdit()
        self.model_textbox.setReadOnly(True)
        hbox = QHBoxLayout()
        hbox.addWidget(self.model_textbox)
        hbox.addWidget(load_button)
        hbox.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.layout().addLayout(hbox)

        # add prediction
        self.prediction_widget = PredictionWidget(
            train_status=self.train_status,
            pred_status=self.pred_status,
            train_signal=None,
            pred_signal=self.pred_config_signal,
        )
        self.prediction_widget.setEnabled(False)
        self.layout().addWidget(self.prediction_widget)

        # changes from the prediction state
        self.pred_status.events.state.connect(self._prediction_state_changed)

    def _select_model_checkpoint(self) -> None:
        """Load a select CAREamics model."""
        selected_file, _filter = QFileDialog.getOpenFileName(
            self, "CAREamics", ".", "CAREamics Model(*.ckpt *.zip)"
        )
        if selected_file is not None and len(selected_file) > 0:
            self.careamist = self._load_model(selected_file)
            if self.careamist is None:
                print(f"Error loading model: {selected_file}")
                return
            self.model_textbox.setText(selected_file)
            self.prediction_widget.setEnabled(True)

    def _load_model(self, model_path: str) -> Optional[CAREamist]:
        """Load a CAREamics model.

        Parameters
        ----------
        model_path : str
            Path to the model checkpoint.

        Returns
        -------
        careamist : CAREamist or None
            CAREamist instance or None if the model could not be loaded.
        """
        try:
            # carefully load the model among the mist: careamist!
            careamist = CAREamist(
                model_path,
                callbacks=[UpdaterCallBack(self._training_queue, self._prediction_queue)]
            )
            # training is already done!
            self.train_status.state = TrainingState.DONE
            self.algo_label.setText(
                f"**Algorithm**: {careamist.cfg.get_algorithm_friendly_name()}"
            )
            self.algo_label.setEnabled(True)

            return careamist

        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def _prediction_state_changed(self, state: PredictionState) -> None:
        """Handle prediction state changes.

        Parameters
        ----------
        state : PredictionState
            New state.
        """
        if state == PredictionState.PREDICTING:
            self.pred_worker = predict_worker(
                self.careamist, self.pred_config_signal, self._prediction_queue
            )

            self.pred_worker.yielded.connect(self._update_from_prediction)
            self.pred_worker.start()

        elif state == PredictionState.STOPPED:
            # this will cause an exception in careamist.predict
            # self.careamist.trainer.predict_loop.teardown()
            # exhaust the data fetcher to stop the prediction
            deque(self.careamist.trainer.predict_loop._data_fetcher, maxlen=0)
            self.careamist.trainer.predict_loop.reset()
            self._prediction_queue.put(
                PredictionUpdate(PredictionUpdateType.SAMPLE_IDX, -1)
            )

    def _update_from_prediction(self, update: PredictionUpdate) -> None:
        """Update the signal from the prediction worker.

        This method receives the updates from the prediction worker.

        Parameters
        ----------
        update : PredictionUpdate
            Update.
        """
        if update.type == PredictionUpdateType.DEBUG:
            print(update.value)
        elif update.type == PredictionUpdateType.EXCEPTION:
            self.pred_status.state = PredictionState.CRASHED

            # print exception without raising it
            print(f"Error: {update.value}")

            if _has_napari:
                ntf.show_error(
                    f"An error occurred during prediction: \n {update.value} \n"
                    f"Note: if you get an error due to the sizes of "
                    f"Tensors, try using tiling."
                )

        else:
            if update.type == PredictionUpdateType.SAMPLE:
                # add image to napari
                # TODO keep scaling?
                if self.viewer is not None:
                    # value is either a numpy array or a list of numpy arrays,
                    # with each sample/time-point as an element
                    if isinstance(update.value, list):
                        # combine all samples
                        samples = np.concatenate(update.value, axis=0)
                    else:
                        samples = update.value

                    # reshape the prediction to match the input axes
                    samples = reshape_prediction(
                        samples, self.careamist.cfg.data_config.axes,
                        self.pred_config_signal.is_3d
                    )

                    self.viewer.add_image(samples, name="Prediction")
            else:
                self.pred_status.update(update)

    def closeEvent(self, event) -> None:
        """Close the plugin.

        Parameters
        ----------
        event : QCloseEvent
            Close event.
        """
        super().closeEvent(event)
        # TODO check training or prediction and stop it


if __name__ == "__main__":
    import faulthandler
    import napari

    log_file_fd = open("fault_log.txt", "a")
    faulthandler.enable(log_file_fd)

    # create a Viewer
    viewer = napari.Viewer()

    # add napari-n2v plugin
    viewer.window.add_dock_widget(PredictionPluginWrapper(viewer))

    # add image to napari
    # viewer.add_image(data[0][0], name=data[0][1]['name'])

    # start UI
    napari.run()
