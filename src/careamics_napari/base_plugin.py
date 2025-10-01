import traceback
from pathlib import Path
from queue import Queue

import numpy as np
from careamics import CAREamist
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QVBoxLayout, QWidget

from careamics_napari.careamics_utils import get_default_n2v_config
from careamics_napari.signals import (
    ExportType,
    PredictionState,
    PredictionStatus,
    PredictionUpdate,
    PredictionUpdateType,
    TrainingState,
    TrainingStatus,
    TrainUpdate,
    TrainUpdateType,
)
from careamics_napari.utils.axes_utils import reshape_prediction
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
from careamics_napari.workers import predict_worker, train_worker

try:
    import napari
    import napari.utils.notifications as ntf
except ImportError:
    _has_napari = False
else:
    _has_napari = True


class BasePlugin(QWidget):
    """CAREamics Base plugin.

    Parameters
    ----------
    napari_viewer : napari.Viewer or None, default=None
        Napari viewer.
    """

    def __init__(
        self,
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
        self.careamist: CAREamist | None = None  # to hold trained careamist
        self.careamist_loaded: CAREamist | None = None  # to hold loaded careamist

        # create statuses, used to keep track of the threads statuses
        self.train_status = TrainingStatus()  # type: ignore
        self.pred_status = PredictionStatus()  # type: ignore

        # create a careamics config (n2v by default)
        self.careamics_config = get_default_n2v_config()

        # create queues, used to communicate between the threads and the UI
        self._training_queue: Queue = Queue(10)
        self._prediction_queue: Queue = Queue(10)

        # changes from the training, prediction or saving state
        self.train_status.events.state.connect(self._training_state_changed)
        self.pred_status.events.state.connect(self._prediction_state_changed)

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

        # calling add_*_ui methods will be happened in sub-classes
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
            careamics_config=self.careamics_config, use_target=use_target
        )
        self.base_layout.addWidget(self.input_data_widget)

    def add_config_ui(self) -> None:
        """Add the training configuration UI to the plugin."""
        self.config_widget = ConfigurationWidget(self.careamics_config)
        self.config_widget.enable_3d_chkbox.clicked.connect(self._set_pred_3d)
        self.config_widget.show_advanced_config.connect(self.show_advanced_config)
        self.base_layout.addWidget(self.config_widget)

    def add_train_button_ui(self) -> None:
        """Add the training button UI to the plugin."""
        self.train_widget = TrainingWidget(self.train_status)
        self.progress_widget = TrainProgressWidget(
            self.careamics_config, self.train_status
        )
        self.base_layout.addWidget(self.train_widget)
        self.base_layout.addWidget(self.progress_widget)

    def add_prediction_ui(self) -> None:
        """Add the prediction UI to the plugin."""
        self.prediction_widget = PredictionWidget(
            self.careamics_config,
            self.train_status,
            self.pred_status,
            self._prediction_queue,
        )
        self.base_layout.addWidget(self.prediction_widget)
        # to get loaded careamist
        self.prediction_widget.careamist_loaded.connect(self._on_careamist_loaded)
        self.prediction_widget.model_from_disk.connect(self._model_selection_changed)

    def add_model_export_ui(self) -> None:
        """Add the model saving UI to the plugin."""
        self.saving_widget = SavingWidget(
            self.careamics_config,
            self.train_status,
        )
        self.saving_widget.export_model.connect(self.export_model)
        self.base_layout.addWidget(self.saving_widget)

    def update_config(self) -> None:
        """Update the configuration from the UI."""
        if self.config_widget is not None:
            self.config_widget.update_config()

        if self.prediction_widget is not None:
            self.prediction_widget.update_config()

        print(f"update_config:\n{self.careamics_config}")

    def export_model(self, destination: Path, export_type: str) -> None:
        """Export the trained model."""
        if self.careamist is None:
            if _has_napari:
                ntf.show_info("No trained model is available for exporting.")
            return

        dims = "3D" if self.careamics_config.is_3D else "2D"
        algo_name = self.careamics_config.algorithm_config.get_algorithm_friendly_name()
        name = f"{algo_name}_{dims}_{self.careamics_config.experiment_name}"

        try:
            if export_type == ExportType.BMZ:
                raise NotImplementedError("Export to BMZ not implemented yet (but soon).")
            else:
                name = name + ".ckpt"
                self.careamist.trainer.save_checkpoint(
                    destination.joinpath(name),
                )
                print(f"Model exported at {destination}")
                if _has_napari:
                    ntf.show_info(f"Model exported at {destination}")

        except Exception as e:
            traceback.print_exc()
            if _has_napari:
                ntf.show_error(str(e))

    def show_advanced_config(self):
        """Show advanced configuration options."""
        raise NotImplementedError("Advanced configuration options are not implemented.")

    def _set_pred_3d(self, state: bool) -> None:
        """Set the 3D mode flag in the prediction widget.

        Parameters
        ----------
        state : bool
            3D mode.
        """
        if self.prediction_widget is not None:
            self.prediction_widget.set_3d(state)

    def _training_state_changed(self, state: TrainingState) -> None:
        """Handle training state changes.

        This includes starting and stopping training.

        Parameters
        ----------
        state : TrainingState
            New state.
        """
        if state == TrainingState.TRAINING:
            # get data sources
            data_sources = self.input_data_widget.get_data_sources()
            if data_sources is None:
                ntf.show_info("Please set the training data first.")
                self.train_status.state = TrainingState.IDLE
                self.train_widget.train_button.setText("Train")
                return

            # update configuration from ui
            self.update_config()
            print(self.careamics_config)

            # start the training thread
            self.train_worker = train_worker(
                self.careamics_config,
                data_sources,
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

        # update prediction widget
        if self.prediction_widget is not None:
            self.prediction_widget.update_button_from_train(state)

    def _prediction_state_changed(self, state: PredictionState) -> None:
        """Handle prediction state changes.

        Parameters
        ----------
        state : PredictionState
            New state.
        """
        # if self.careamist is None and self.careamist_loaded is None:
        #     ntf.show_info("No trained or loaded model is available for prediction.")
        #     self.pred_status.state = PredictionState.STOPPED
        #     return
        careamist = self._which_careamist()
        if careamist is None:
            self.pred_status.state = PredictionState.STOPPED
            return

        if state == PredictionState.PREDICTING:
            # get the prediction data
            data_source = self.prediction_widget.get_data_source()
            if data_source is None:
                ntf.show_info("Please set the prediction data first.")
                self.pred_status.state = PredictionState.IDLE
                self.prediction_widget.predict_button.setText("Predict")
                return

            # update configuration from ui
            self.update_config()

            # start the prediction thread
            self.pred_worker = predict_worker(
                careamist,
                data_source,
                self.careamics_config,
                self._prediction_queue,
            )
            self.pred_worker.yielded.connect(self._update_from_prediction)
            self.pred_worker.start()

        elif state == PredictionState.STOPPED:
            # prediction stopped: reset the progress bar
            self._prediction_queue.put(
                PredictionUpdate(PredictionUpdateType.SAMPLE_IDX, -1)
            )

    def _on_careamist_loaded(self, careamist: CAREamist) -> None:
        """Event handler called when a CAREamics instance has been loaded."""
        self.careamist_loaded = careamist
        print(
            f"CAREamics instance loaded: "
            f"{self.careamist_loaded.cfg.get_algorithm_friendly_name()}"
        )

    def _model_selection_changed(self, from_disk: bool) -> None:
        """Event handler called when user changed the model selection."""
        # update the prediction and stop buttons
        if not from_disk:
            self.prediction_widget.update_button_from_train(self.train_status.state)
        elif self.careamist_loaded is not None:
            self.prediction_widget.predict_button.setEnabled(True)
            self.prediction_widget.stop_button.setEnabled(False)

    def _which_careamist(self) -> CAREamist | None:
        """Which careamist to use? Trained one or the loaded one."""
        # if load from disk option is selected
        if self.prediction_widget.load_from_disk:
            careamist = self.careamist_loaded
            if careamist is None:
                ntf.show_warning("No model was loaded from disk!")
        else:
            careamist = self.careamist
            if careamist is None:
                ntf.show_warning("No trained model is available.")

        return careamist

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
                        samples,  # type: ignore
                        self.careamics_config.data_config.axes,  # type: ignore
                        self.careamics_config.is_3D,
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
    import napari

    # create a Viewer
    viewer = napari.Viewer()

    base_plugin = BasePlugin(viewer)
    base_plugin.add_careamics_banner()
    base_plugin.add_train_input_ui(base_plugin.careamics_config.needs_gt)
    base_plugin.add_config_ui()
    base_plugin.add_train_button_ui()
    base_plugin.add_prediction_ui()
    base_plugin.add_model_export_ui()
    viewer.window.add_dock_widget(base_plugin)
    # add image to napari
    # viewer.add_image(data[0][0], name=data[0][1]['name'])
    # start UI
    napari.run()
