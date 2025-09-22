from collections import deque
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING

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
)
from careamics_napari.widgets import (
    CAREamicsBanner,
    ConfigurationWidget,
    PredictionWidget,
    SavingWidget,
    ScrollWidgetWrapper,
    TrainDataWidget,
    TrainingWidget,
    TrainProgressWidget,
    create_gpu_label,
)
from careamics_napari.workers import predict_worker, save_worker, train_worker
from careamics_napari.careamics_utils import get_default_n2v_config
from careamics_napari.utils.axes_utils import reshape_prediction

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


class ScrollPluginWrapper(ScrollWidgetWrapper):
    """Wrap a plugin widget within a scrolling wrapper."""

    def __init__(
        self: Self,
        plugin: QWidget,
    ) -> None:
        """Initialize the plugin.

        Parameters
        ----------
        plugin : QWidget
            Plugin widget to wrap.
        """
        super().__init__(plugin)


class BasePlugin(QWidget):
    """CAREamics Base plugin.

    Parameters
    ----------
    napari_viewer : napari.Viewer or None, default=None
        Napari viewer.
    """

    def __init__(
        self: Self,
        napari_viewer: napari.Viewer | None = None,
    ) -> None:
        """Initialize the plugin.

        Parameters
        ----------
        napari_viewer : napari.Viewer or None, default=None
            Napari viewer.
        """
        super().__init__()
        self.viewer = napari_viewer
        self.careamist: CAREamist | None = None

        # create statuses, used to keep track of the threads statuses
        self.train_status = TrainingStatus()  # type: ignore
        self.pred_status = PredictionStatus()  # type: ignore
        self.save_status = SavingStatus()  # type: ignore

        # create a careamics config (we just use the common params)
        self.careamics_config = get_default_n2v_config()

        # create signals, used to hold the various parameters modified by the UI
        self.pred_config_signal = PredictionSignal()
        self.save_config_signal = SavingSignal()
        # make sure that the prediction 3D mode is the same as the training one
        # self.train_config_signal.events.is_3d.connect(self._set_pred_3d)

        # create queues, used to communicate between the threads and the UI
        self._training_queue: Queue = Queue(10)
        self._prediction_queue: Queue = Queue(10)

        # changes from the training, prediction or saving state
        self.train_status.events.state.connect(self._training_state_changed)
        self.pred_status.events.state.connect(self._prediction_state_changed)
        self.save_status.events.state.connect(self._saving_state_changed)

        # main layout
        self.base_layout = QVBoxLayout()
        # scrolling content
        scroll_content = QWidget()
        scroll_content.setLayout(self.base_layout)
        scroll = ScrollWidgetWrapper(scroll_content)
        vbox = QVBoxLayout()
        vbox.addWidget(scroll)
        self.setLayout(vbox)
        self.setMinimumWidth(200)
        # calling add ui methods will happen in the child classes
        # to allow more flexibility while saving some code duplication.

    def add_careamics_banner(self, desc: str = "") -> None:
        """Add the CAREamics banner and GPU label to the plugin."""
        if len(desc) == 0:
            desc = "CAREamics UI for training denoising models."
        self.base_layout.addWidget(
            CAREamicsBanner(
                title="CAREamics",
                short_desc=(desc),
            )
        )
        # GPU label
        gpu_button = create_gpu_label()
        gpu_button.setAlignment(Qt.AlignmentFlag.AlignRight)
        gpu_button.setContentsMargins(0, 5, 0, 0)  # top margin
        self.base_layout.addWidget(gpu_button)

    def add_train_input_ui(self, use_target: bool = False) -> None:
        """Add the train input data selection UI to the plugin."""
        self.input_data_widget = TrainDataWidget(
            signal=self.careamics_config,
            use_target=use_target
        )
        self.base_layout.addWidget(self.input_data_widget)

    def add_config_ui(self) -> None:
        """Add the training configuration UI to the plugin."""
        self.config_widget = ConfigurationWidget(self.careamics_config)
        self.base_layout.addWidget(self.config_widget)

    def add_train_button_ui(self) -> None:
        """Add the training button UI to the plugin."""
        self.train_widget = TrainingWidget(self.train_status)
        self.progress_widget = TrainProgressWidget(
            self.train_status, self.careamics_config
        )
        self.base_layout.addWidget(self.train_widget)
        self.base_layout.addWidget(self.progress_widget)

    def add_prediction_ui(self) -> None:
        """Add the prediction UI to the plugin."""
        self.prediction_widget = PredictionWidget(
            self.train_status,
            self.pred_status,
            self.careamics_config,
            self.pred_config_signal,
        )
        self.base_layout.addWidget(self.prediction_widget)

    def add_model_saving_ui(self) -> None:
        """Add the model saving UI to the plugin."""
        self.saving_widget = SavingWidget(
            train_status=self.train_status,
            save_status=self.save_status,
            save_signal=self.save_config_signal,
        )
        self.base_layout.addWidget(self.saving_widget)

    def _set_pred_3d(self, is_3d: bool) -> None:
        """Set the 3D mode flag in the prediction signal.

        Parameters
        ----------
        is_3d : bool
            3D mode.
        """
        self.pred_config_signal.is_3d = is_3d

    def _training_state_changed(self, state: TrainingState) -> None:
        """Handle training state changes.

        This includes starting and stopping training.

        Parameters
        ----------
        state : TrainingState
            New state.
        """
        if (
            len(self.careamics_config.path_train) == 0
            and self.careamics_config.layer_train is None
        ):
            ntf.show_info("Please set the training data first.")
            self.train_status.state = TrainingState.STOPPED
            self.train_widget.train_button.setText("Train")
            return

        if state == TrainingState.TRAINING:
            self.train_worker = train_worker(
                self.careamics_config,
                self._training_queue,
                self._prediction_queue,
                self.careamist,
            )

            self.train_worker.yielded.connect(self._update_from_training)
            self.train_worker.start()

        elif state == TrainingState.STOPPED:
            if self.careamist is not None:
                self.careamist.stop_training()

        elif state == TrainingState.CRASHED or state == TrainingState.IDLE:
            del self.careamist
            self.careamist = None

    def _prediction_state_changed(self, state: PredictionState) -> None:
        """Handle prediction state changes.

        Parameters
        ----------
        state : PredictionState
            New state.
        """
        if self.careamist is None:
            ntf.show_info("No trained model is available for prediction.")
            self.pred_status.state = PredictionState.STOPPED
            return

        if state == PredictionState.PREDICTING:
            self.pred_worker = predict_worker(
                self.careamist, self.pred_config_signal, self._prediction_queue
            )

            self.pred_worker.yielded.connect(self._update_from_prediction)
            self.pred_worker.start()

        elif state == PredictionState.STOPPED:
            # exhaust the data fetcher to stop the prediction
            if self.careamist.trainer.predict_loop._data_fetcher is not None:
                deque(self.careamist.trainer.predict_loop._data_fetcher, maxlen=0)
                self.careamist.trainer.predict_loop.reset()
                self._prediction_queue.put(
                    PredictionUpdate(PredictionUpdateType.SAMPLE_IDX, -1)
                )

    def _saving_state_changed(self, state: SavingState) -> None:
        """Handle saving state changes.

        Parameters
        ----------
        state : SavingState
            New state.
        """
        if state == SavingState.SAVING and self.careamist is not None:
            self.save_worker = save_worker(
                self.careamist, self.careamics_config, self.save_config_signal
            )
            self.save_worker.yielded.connect(self._update_from_saving)
            self.save_worker.start()

    def _update_from_training(self, update: TrainUpdate) -> None:
        """Update the training status from the training worker.

        This method receives the updates from the training worker.

        Parameters
        ----------
        update : TrainUpdate
            Update.
        """
        if update.type == TrainUpdateType.CAREAMIST:
            if isinstance(update.value, CAREamist):
                self.careamist = update.value
        elif update.type == TrainUpdateType.DEBUG:
            print(update.value)
        elif update.type == TrainUpdateType.EXCEPTION:
            self.train_status.state = TrainingState.CRASHED

            if isinstance(update.value, Exception):
                raise update.value
        else:
            self.train_status.update(update)

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
                    # value is either a numpy array or
                    # a list of numpy arrays with each sample/time-point as an element
                    if isinstance(update.value, list):
                        # combine all samples
                        samples = np.concatenate(update.value, axis=0)
                    else:
                        samples = update.value

                    # reshape the prediction to match the input axes
                    samples = reshape_prediction(
                        samples,
                        self.careamics_config.axes,
                        self.pred_config_signal.is_3d
                    )
                    self.viewer.add_image(samples, name="Prediction")
            else:
                self.pred_status.update(update)

    def _update_from_saving(self, update: SavingUpdate) -> None:
        """Update the signal from the saving worker.

        This method receives the updates from the saving worker.

        Parameters
        ----------
        update : SavingUpdate
            Update.
        """
        if update.type == SavingUpdateType.DEBUG:
            print(update.value)
        elif update.type == SavingUpdateType.EXCEPTION:
            self.save_status.state = SavingState.CRASHED

            if _has_napari:
                ntf.show_error(f"An error occurred during saving: \n {update.value}")

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
    # viewer.window.add_dock_widget(
    #     ScrollPluginWrapper(BasePlugin(viewer))
    # )
    base_plugin = BasePlugin(viewer)
    base_plugin.add_careamics_banner()
    base_plugin.add_config_ui()
    viewer.window.add_dock_widget(base_plugin)
    # add image to napari
    # viewer.add_image(data[0][0], name=data[0][1]['name'])
    # start UI
    napari.run()
