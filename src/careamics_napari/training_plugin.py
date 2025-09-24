"""CAREamics training Qt widget."""

from collections import deque
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Optional

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
    TrainingSignal,
    TrainingState,
    TrainingStatus,
    TrainUpdate,
    TrainUpdateType,
)
from careamics_napari.widgets import (
    AlgorithmSelectionWidget,
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

# TODO: add logging to napari
# TODO add loading of existing model

# TODO current issues:
# - cannot restart prediction after exception
# - random illegal hardware instruction
# - saving BMZ
# - loading trained model
# - prediction only plugin (and to disk)

# TODO prediction should reuse the name of the layer


class TrainPluginWrapper(ScrollWidgetWrapper):
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
        super().__init__(TrainPlugin(napari_viewer))


class TrainPlugin(QWidget):
    """CAREamics training plugin.

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
        self.train_status = TrainingStatus()  # type: ignore
        self.pred_status = PredictionStatus()  # type: ignore
        self.save_status = SavingStatus()  # type: ignore

        # create signals, used to hold the various parameters modified by the UI
        self.train_config_signal = TrainingSignal()  # type: ignore
        self.pred_config_signal = PredictionSignal()
        self.save_config_signal = SavingSignal()

        self.train_config_signal.events.is_3d.connect(self._set_pred_3d)

        # create queues, used to communicate between the threads and the UI
        self._training_queue: Queue = Queue(10)
        self._prediction_queue: Queue = Queue(10)

        # set workdir
        self.train_config_signal.work_dir = Path.cwd()

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

        # add GPU label and algorithm selection
        algo_panel = QWidget()
        algo_panel.setLayout(QHBoxLayout())

        gpu_button = create_gpu_label()
        gpu_button.setAlignment(Qt.AlignmentFlag.AlignRight)
        gpu_button.setContentsMargins(0, 5, 0, 0)  # top margin

        algo_choice = AlgorithmSelectionWidget(training_signal=self.train_config_signal)
        gpu_button.setAlignment(Qt.AlignmentFlag.AlignLeft)

        algo_panel.layout().addWidget(algo_choice)
        algo_panel.layout().addWidget(gpu_button)

        self.layout().addWidget(algo_panel)

        # add data tabs
        self.data_stck = QStackedWidget()
        self.data_layers = [
            TrainDataWidget(signal=self.train_config_signal),
            TrainDataWidget(signal=self.train_config_signal, use_target=True),
        ]
        for layer in self.data_layers:
            self.data_stck.addWidget(layer)
        self.data_stck.setCurrentIndex(0)

        self.layout().addWidget(self.data_stck)

        # add configuration widget
        self.config_widget = ConfigurationWidget(self.train_config_signal)
        self.layout().addWidget(self.config_widget)

        # add train widget
        self.train_widget = TrainingWidget(self.train_status)
        self.layout().addWidget(self.train_widget)

        # add progress widget
        self.progress_widget = TrainProgressWidget(
            self.train_status, self.train_config_signal
        )
        self.layout().addWidget(self.progress_widget)

        # add prediction
        self.prediction_widget = PredictionWidget(
            self.train_status,
            self.pred_status,
            self.train_config_signal,
            self.pred_config_signal,
        )
        self.layout().addWidget(self.prediction_widget)

        # add saving
        self.saving_widget = SavingWidget(
            train_status=self.train_status,
            save_status=self.save_status,
            save_signal=self.save_config_signal,
        )
        self.layout().addWidget(self.saving_widget)

        # connect signals
        # changes from the selected algorithm
        self.train_config_signal.events.algorithm.connect(self._set_data_from_algorithm)
        self._set_data_from_algorithm(
            self.train_config_signal.algorithm
        )  # force update

        # changes from the training, prediction or saving state
        self.train_status.events.state.connect(self._training_state_changed)
        self.pred_status.events.state.connect(self._prediction_state_changed)
        self.save_status.events.state.connect(self._saving_state_changed)

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
        if state == TrainingState.TRAINING:
            self.train_worker = train_worker(
                self.train_config_signal,
                self._training_queue,
                self._prediction_queue,
                self.careamist,
                self.pred_status,
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
        if state == PredictionState.PREDICTING:
            self.pred_worker = predict_worker(
                self.careamist, self.pred_config_signal, self._prediction_queue
            )

            self.pred_worker.yielded.connect(self._update_from_prediction)
            self.pred_worker.start()

        elif state == PredictionState.STOPPED:
            pass

    def _saving_state_changed(self, state: SavingState) -> None:
        """Handle saving state changes.

        Parameters
        ----------
        state : SavingState
            New state.
        """
        if state == SavingState.SAVING:
            self.save_worker = save_worker(
                self.careamist, self.train_config_signal, self.save_config_signal
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
                    # value is eighter a numpy array or a list of numpy arrays with each sample/timepoint as an element
                    if isinstance(update.value, list):
                        # combine all samples
                        samples = np.concatenate(update.value, axis=0)
                    else:
                        samples = update.value
                    
                    # reshape the prediction to match the input axes
                    samples = reshape_prediction(samples, self.train_config_signal.axes, self.pred_config_signal.is_3d)

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

    def _set_data_from_algorithm(self, name: str) -> None:
        """Update the data selection widget based on the algorithm.

        Parameters
        ----------
        name : str
            Algorithm name.
        """
        if (
            name == SupportedAlgorithm.CARE.value
            or name == SupportedAlgorithm.N2N.value
        ):
            self.data_stck.setCurrentIndex(1)
        else:
            self.data_stck.setCurrentIndex(0)

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
    # from qtpy.QtWidgets import QApplication
    # import sys

    # # Create a QApplication instance
    # app = QApplication(sys.argv)

    # # Instantiate widget
    # widget = TrainPluginWrapper()

    # # Show the widget
    # widget.show()

    # # Run the application event loop
    # sys.exit(app.exec_())
    import faulthandler

    import napari

    log_file_fd = open("fault_log.txt", "a")
    faulthandler.enable(log_file_fd)

    # create a Viewer
    viewer = napari.Viewer()

    # add napari-n2v plugin
    viewer.window.add_dock_widget(TrainPluginWrapper(viewer))

    # add image to napari
    # viewer.add_image(data[0][0], name=data[0][1]['name'])

    # start UI
    napari.run()
