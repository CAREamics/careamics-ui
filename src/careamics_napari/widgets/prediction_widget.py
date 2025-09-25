"""Widget used to run prediction from the Training plugin."""

from queue import Queue

import napari.utils.notifications as ntf
import numpy as np
from careamics import CAREamist
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from careamics_napari.careamics_utils import BaseConfig, UpdaterCallBack
from careamics_napari.signals import (
    PredictionState,
    PredictionStatus,
    TrainingState,
    TrainingStatus,
)
from careamics_napari.widgets.predict_data_widget import PredictDataWidget
from careamics_napari.widgets.qt_widgets import (
    PowerOfTwoSpinBox,
    create_int_spinbox,
    create_progressbar,
)
from careamics_napari.widgets.utils import bind


class PredictionWidget(QGroupBox):
    """A widget to run prediction on images from within the Training plugin.

    Parameters
    ----------
    careamics_config : BaseConfig
            The configuration for the CAREamics algorithm.
    train_status : TrainingStatus or None, default=None
        The training status signal.
    pred_status : PredictionStatus or None, default=None
        The prediction status signal.
    """

    # set a signal to send a careamist object
    # when it's loaded from disk.
    careamist_loaded = Signal(CAREamist)

    def __init__(
        self,
        careamics_config: BaseConfig,
        train_status: TrainingStatus | None = None,
        pred_status: PredictionStatus | None = None,
        prediction_queue: Queue | None = None,
    ) -> None:
        """Initialize the widget.

        Parameters
        ----------
        careamics_config : BaseConfig
            The configuration for the CAREamics algorithm.
        train_status : TrainingStatus or None, default=None
            The training status signal.
        pred_status : PredictionStatus or None, default=None
            The prediction status signal.
        prediction_queue : Queue or None, default=None
            The prediction queue.
        """
        super().__init__()

        self.configuration = careamics_config
        self.train_status = (
            TrainingStatus() if train_status is None else train_status  # type: ignore
        )
        self.pred_status = (
            PredictionStatus() if pred_status is None else pred_status  # type: ignore
        )
        self.prediction_queue = (
            Queue(10) if prediction_queue is None else prediction_queue
        )

        self.setTitle("Prediction")

        # model selection
        self.from_train_radiobutton = QRadioButton("From the trained model")
        self.from_train_radiobutton.setChecked(True)
        self.from_disk_radiobutton = QRadioButton("Load model from disk")
        self.model_textbox = QLineEdit()
        self.model_textbox.setReadOnly(True)
        self.model_textbox.setEnabled(False)
        self.load_button = QPushButton("Load...")
        self.load_button.setEnabled(False)

        # data selection
        self.predict_data_widget = PredictDataWidget()

        # checkbox
        self.tiling_cbox = QCheckBox("Tile prediction")
        self.tiling_cbox.setChecked(True)
        self.tiling_cbox.setToolTip(
            "Select to predict the image by tiles, allowing to predict on large images."
        )

        # tiling spinboxes
        self.tile_size_xy_spin = PowerOfTwoSpinBox(64, 1024, 64)
        self.tile_size_xy_spin.setToolTip("Tile size in the xy dimension.")
        # self.tile_size_xy.setEnabled(False)

        self.tile_size_z_spin = PowerOfTwoSpinBox(4, 32, 8)
        self.tile_size_z_spin.setToolTip("Tile size in the z dimension.")
        self.tile_size_z_spin.setEnabled(self.configuration.is_3D)

        # batch size spinbox
        self.batch_size_spin = create_int_spinbox(1, 512, 1, 1)
        self.batch_size_spin.setToolTip(
            "Number of patches per batch (decrease if GPU memory is insufficient)"
        )
        # self.batch_size_spin.setEnabled(False)

        # prediction progress bar
        self.pb_prediction = create_progressbar(
            max_value=20, text_format="Prediction ?/?"
        )
        self.pb_prediction.setToolTip("Show the progress of the prediction")

        # predict button
        self.predict_button = QPushButton("Predict", self)
        self.predict_button.setMinimumWidth(120)
        self.predict_button.setEnabled(False)
        self.predict_button.setToolTip("Run the trained model on the images")
        # stop button
        self.stop_button = QPushButton("Stop", self)
        self.stop_button.setMinimumWidth(120)
        self.stop_button.setEnabled(False)
        self.stop_button.setToolTip("Stop the prediction")

        # layout
        vbox = QVBoxLayout()
        model_vbox = QVBoxLayout()
        model_vbox.addWidget(self.from_train_radiobutton)
        model_vbox.addWidget(self.from_disk_radiobutton)
        hbox = QHBoxLayout()
        hbox.addWidget(self.model_textbox)
        hbox.addWidget(self.load_button)
        model_vbox.addLayout(hbox)
        vbox.addLayout(model_vbox)
        vbox.addWidget(self.predict_data_widget)
        vbox.addWidget(self.tiling_cbox)
        tiling_form = QFormLayout()
        tiling_form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)  # type: ignore
        tiling_form.setFieldGrowthPolicy(
            QFormLayout.AllNonFixedFieldsGrow  # type: ignore
        )
        tiling_form.addRow("XY tile size", self.tile_size_xy_spin)
        tiling_form.addRow("Z tile size", self.tile_size_z_spin)
        tiling_form.addRow("Batch size", self.batch_size_spin)
        vbox.addLayout(tiling_form)
        vbox.addWidget(self.pb_prediction)
        hbox = QHBoxLayout()
        hbox.addWidget(self.predict_button, alignment=Qt.AlignLeft)  # type: ignore
        hbox.addWidget(self.stop_button, alignment=Qt.AlignRight)  # type: ignore
        vbox.addLayout(hbox)
        self.setLayout(vbox)

        # actions
        self.from_train_radiobutton.clicked.connect(self._model_selection_changed)
        self.from_disk_radiobutton.clicked.connect(self._model_selection_changed)
        self.load_button.clicked.connect(self._select_model_checkpoint)
        self.tiling_cbox.clicked.connect(self._update_tilings)
        self.predict_button.clicked.connect(self._predict_button_clicked)
        self.stop_button.clicked.connect(self._stop_button_clicked)

        self.pred_status.events.state.connect(self._update_button_from_pred)
        self.pred_status.events.sample_idx.connect(self._update_sample_idx)
        self.pred_status.events.max_samples.connect(self._update_max_sample)

        # bind properties
        self._bind_properties()

    def set_3d(self, state: bool) -> None:
        """Enable the z tile size spinbox if the data is 3D.

        Parameters
        ----------
        state : bool
            The new state of the 3D checkbox.
        """
        # this method can be used by the parent plugin when the train config is updated.
        self.configuration.is_3D = state
        self.tile_size_z_spin.setEnabled(self.do_tiling and state)

    def update_button_from_train(self, state: TrainingState) -> None:
        """Update the predict button based on the training state.

        Parameters
        ----------
        state : TrainingState
            The new state of the training plugin.
        """
        if state == TrainingState.DONE:
            self.predict_button.setEnabled(True)
        else:
            self.predict_button.setEnabled(False)

    def get_data_source(self) -> str | np.ndarray | None:
        """Get the selected data sources from the predict data widget."""
        return self.predict_data_widget.get_data_sources()

    def update_config(self) -> None:
        """Update the prediction configuration from the UI element."""
        # tile size
        self.configuration.tile_size = None
        if self.do_tiling:
            _tile_size = [self.tile_size_xy, self.tile_size_xy]
            if self.configuration.is_3D:
                _tile_size.insert(0, self.tile_size_z)
            self.configuration.tile_size = tuple(_tile_size)

        # batch size
        self.configuration.pred_batch_size = self.batch_size

    def _bind_properties(self) -> None:
        """Create and bind the properties to the UI elements."""
        # tiling
        type(self).do_tiling = bind(self.tiling_cbox, "checked", True)
        # tile size in xy
        type(self).tile_size_xy = bind(self.tile_size_xy_spin, "value", 64)
        # tile size in z
        type(self).tile_size_z = bind(self.tile_size_z_spin, "value", 8)
        # batch size
        type(self).batch_size = bind(self.batch_size_spin, "value", 1)

    def _model_selection_changed(self) -> None:
        """Update model selection ui."""
        load_from_disk = self.from_disk_radiobutton.isChecked()
        self.model_textbox.setEnabled(load_from_disk)
        self.load_button.setEnabled(load_from_disk)

    def _select_model_checkpoint(self) -> None:
        """Load a selected CAREamics model."""
        selected_file, _filter = QFileDialog.getOpenFileName(
            self, "CAREamics", ".", "CAREamics Model(*.ckpt *.zip)"
        )
        if selected_file is not None and len(selected_file) > 0:
            careamist = self._load_model(selected_file)
            if careamist is None:
                ntf.show_error(f"Error loading the model: {selected_file}")
                return
            # sent the careamist to the parent window / plugin
            self.careamist_loaded.emit(careamist)
            self.model_textbox.setText(selected_file)
            self.predict_button.setEnabled(True)

    def _load_model(self, model_path: str) -> CAREamist | None:
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
                work_dir=self.configuration.work_dir,
                callbacks=[UpdaterCallBack(self.prediction_queue)],
            )
            # training is already done!
            # self.train_status.state = TrainingState.DONE

            # check the loaded model algorithm
            # to be compatible with the current configuration
            model_algo = careamist.cfg.get_algorithm_friendly_name()
            config_algo = self.configuration.get_algorithm_friendly_name()
            if model_algo != config_algo:
                raise ValueError(
                    f"The loaded model ({model_algo}) does not match "
                    f"the current configuration ({config_algo})."
                )

            return careamist

        except Exception as e:
            print(f"Error loading the model:\n{e}")
            return None

    def _update_tilings(self, state: bool) -> None:
        """Update the widgets and the signal tiling parameter.

        Parameters
        ----------
        state : bool
            The new state of the tiling checkbox.
        """
        # self.do_tiling = state
        self.tile_size_xy_spin.setEnabled(state)
        self.batch_size_spin.setEnabled(state)
        self.tile_size_z_spin.setEnabled(state and self.configuration.is_3D)

    def _update_3d_tiles(self, state: bool) -> None:
        """Enable the z tile size spinbox if the data is 3D and tiled.

        Parameters
        ----------
        state : bool
            The new state of the 3D checkbox.
        """
        if self.pred_signal.tiled:
            self.tile_size_z_spin.setEnabled(state)

    def _update_max_sample(self, max_sample: int) -> None:
        """Update the maximum value of the progress bar.

        Parameters
        ----------
        max_sample : int
            The new maximum value of the progress bar.
        """
        self.pb_prediction.setMaximum(max_sample)

    def _update_sample_idx(self, sample: int) -> None:
        """Update the value of the progress bar.

        Parameters
        ----------
        sample : int
            The new value of the progress bar.
        """
        self.pb_prediction.setValue(sample + 1)
        self.pb_prediction.setFormat(
            f"Sample {sample + 1}/{self.pred_status.max_samples}"
        )

    def _predict_button_clicked(self) -> None:
        """Run the prediction on the images."""
        if self.pred_status is not None:
            if (
                self.pred_status.state == PredictionState.IDLE
                or self.train_status.state == TrainingState.DONE
                or self.pred_status.state == PredictionState.CRASHED
            ):
                self.predict_button.setEnabled(False)
                self.stop_button.setEnabled(True)
                self.pred_status.state = PredictionState.PREDICTING

    def _stop_button_clicked(self) -> None:
        """Stop the prediction."""
        if self.pred_status.state == PredictionState.PREDICTING:
            self.stop_button.setEnabled(False)
            self.pred_status.state = PredictionState.STOPPED

    def _update_button_from_pred(self, state: PredictionState) -> None:
        """Update the predict button based on the prediction state.

        Parameters
        ----------
        state : PredictionState
            The new state of the prediction plugin.
        """
        if (
            state == PredictionState.DONE
            or state == PredictionState.CRASHED
            or state == PredictionState.STOPPED
        ):
            self.predict_button.setEnabled(True)


if __name__ == "__main__":
    # import sys
    import napari

    # from qtpy.QtWidgets import QApplication
    from careamics_napari.careamics_utils import get_default_n2v_config

    config = get_default_n2v_config()
    train_status = TrainingStatus()  # type: ignore
    pred_status = PredictionStatus()  # type: ignore

    # create a Viewer
    viewer = napari.Viewer()

    # Create a QApplication instance
    # app = QApplication(sys.argv)
    widget = PredictionWidget(config, train_status, pred_status)
    # widget.show()

    viewer.window.add_dock_widget(widget)
    napari.run()
    # Run the application event loop
    # sys.exit(app.exec_())
