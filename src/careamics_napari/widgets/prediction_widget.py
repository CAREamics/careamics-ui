"""Widget used to run prediction from the Training plugin."""

from typing import Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from typing_extensions import Self

from careamics_napari.signals import (
    PredictionSignal,
    PredictionState,
    PredictionStatus,
    TrainingSignal,
    TrainingState,
    TrainingStatus,
)

from .predict_data_widget import PredictDataWidget
from .qt_widgets import PowerOfTwoSpinBox, create_int_spinbox, create_progressbar


class PredictionWidget(QGroupBox):
    """A widget to run prediction on images from within the Training plugin.

    Parameters
    ----------
    train_status : TrainingStatus or None, default=None
        The training status signal.
    pred_status : PredictionStatus or None, default=None
        The prediction status signal.
    train_signal : TrainingSignal or None, default=None
        The training configuration signal.
    pred_signal : PredictionSignal or None, default=None
        The prediction configuration signal.
    """

    def __init__(
        self: Self,
        train_status: Optional[TrainingStatus] = None,
        pred_status: Optional[PredictionStatus] = None,
        train_signal: Optional[TrainingSignal] = None,
        pred_signal: Optional[PredictionSignal] = None,
    ) -> None:
        """Initialize the widget.

        Parameters
        ----------
        train_status : TrainingStatus or None, default=None
            The training status signal.
        pred_status : PredictionStatus or None, default=None
            The prediction status signal.
        train_signal : TrainingSignal or None, default=None
            The training configuration signal.
        pred_signal : PredictionSignal or None, default=None
            The prediction configuration signal.
        """
        super().__init__()

        self.train_status = (
            TrainingStatus() if train_status is None else train_status  # type: ignore
        )
        self.pred_status = (
            PredictionStatus() if pred_status is None else pred_status  # type: ignore
        )
        self.train_signal = (
            TrainingSignal() if train_signal is None else train_signal  # type: ignore
        )
        self.pred_signal = PredictionSignal() if pred_signal is None else pred_signal

        self.setTitle("Prediction")
        self.setLayout(QVBoxLayout())

        # data selection
        predict_data_widget = PredictDataWidget(self.pred_signal)
        self.layout().addWidget(predict_data_widget)

        # checkbox
        self.tiling_cbox = QCheckBox("Tile prediction")
        self.tiling_cbox.setToolTip(
            "Select to predict the image by tiles, allowing "
            "to predict on large images."
        )
        self.layout().addWidget(self.tiling_cbox)

        # tiling spinboxes
        self.tile_size_xy = PowerOfTwoSpinBox(64, 1024, self.pred_signal.tile_size_xy)
        self.tile_size_xy.setToolTip("Tile size in the xy dimension.")
        self.tile_size_xy.setEnabled(False)

        self.tile_size_z = PowerOfTwoSpinBox(4, 32, self.pred_signal.tile_size_z)
        self.tile_size_z.setToolTip("Tile size in the z dimension.")
        self.tile_size_z.setEnabled(False)

        self.batch_size_spin = create_int_spinbox(1, 512, 1, 1)
        self.batch_size_spin.setToolTip(
            "Number of patches per batch (decrease if GPU memory is insufficient)"
        )
        self.batch_size_spin.setEnabled(False)

        tiling_form = QFormLayout()
        tiling_form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        tiling_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        tiling_form.addRow("XY tile size", self.tile_size_xy)
        tiling_form.addRow("Z tile size", self.tile_size_z)
        tiling_form.addRow("Batch size", self.batch_size_spin)
        tiling_widget = QWidget()
        tiling_widget.setLayout(tiling_form)
        self.layout().addWidget(tiling_widget)

        # prediction progress bar
        self.pb_prediction = create_progressbar(
            max_value=20, text_format="Prediction ?/?"
        )
        self.pb_prediction.setToolTip("Show the progress of the prediction")

        # predict button
        predictions = QWidget()
        predictions.setLayout(QHBoxLayout())
        self.predict_button = QPushButton("Predict", self)
        self.predict_button.setMinimumWidth(120)
        self.predict_button.setEnabled(False)
        self.predict_button.setToolTip("Run the trained model on the images")
        # stop button
        self.stop_button = QPushButton("Stop", self)
        self.stop_button.setMinimumWidth(120)
        self.stop_button.setEnabled(False)
        self.stop_button.setToolTip("Stop the prediction")

        predictions.layout().addWidget(self.predict_button, alignment=Qt.AlignLeft)
        predictions.layout().addWidget(self.stop_button, alignment=Qt.AlignRight)

        # add to the group
        self.layout().addWidget(self.pb_prediction)
        self.layout().addWidget(predictions)

        # actions
        self.tiling_cbox.stateChanged.connect(self._update_tiles)

        if self.pred_status is not None and self.train_status is not None:
            # what to do when the buttons are clicked
            self.predict_button.clicked.connect(self._predict_button_clicked)
            self.stop_button.clicked.connect(self._stop_button_clicked)

            self.tile_size_xy.valueChanged.connect(self._set_xy_tile_size)
            self.tile_size_z.valueChanged.connect(self._set_z_tile_size)
            self.batch_size_spin.valueChanged.connect(self._set_batch_size)

            # listening to the signals
            self.train_signal.events.is_3d.connect(self._set_3d)
            self.train_status.events.state.connect(self._update_button_from_train)
            self.pred_status.events.state.connect(self._update_button_from_pred)

            self.pred_status.events.sample_idx.connect(self._update_sample_idx)
            self.pred_status.events.max_samples.connect(self._update_max_sample)

    def _set_xy_tile_size(self: Self, size: int) -> None:
        """Update the signal tile size in the xy dimension.

        Parameters
        ----------
        size : int
            The new tile size in the xy dimension.
        """
        if self.pred_signal is not None:
            self.pred_signal.tile_size_xy = size

    def _set_z_tile_size(self: Self, size: int) -> None:
        """Update the signal tile size in the z dimension.

        Parameters
        ----------
        size : int
            The new tile size in the z dimension.
        """
        if self.pred_signal is not None:
            self.pred_signal.tile_size_z = size

    def _set_batch_size(self: Self, size: int) -> None:
        """Update the signal batch size.

        Parameters
        ----------
        size : int
            The new batch size.
        """
        if self.pred_signal is not None:
            self.pred_signal.batch_size = size

    def _set_3d(self: Self, state: bool) -> None:
        """Enable the z tile size spinbox if the data is 3D.

        Parameters
        ----------
        state : bool
            The new state of the 3D checkbox.
        """
        if self.pred_signal.tiled:
            self.tile_size_z.setEnabled(state)

    def _update_tiles(self: Self, state: bool) -> None:
        """Update the weidgets and the signal tiling parameter.

        Parameters
        ----------
        state : bool
            The new state of the tiling checkbox.
        """
        self.pred_signal.tiled = state
        self.tile_size_xy.setEnabled(state)
        self.batch_size_spin.setEnabled(state)

        if self.train_signal.is_3d:
            self.tile_size_z.setEnabled(state)

    def _update_3d_tiles(self: Self, state: bool) -> None:
        """Enable the z tile size spinbox if the data is 3D and tiled.

        Parameters
        ----------
        state : bool
            The new state of the 3D checkbox.
        """
        if self.pred_signal.tiled:
            self.tile_size_z.setEnabled(state)

    def _update_max_sample(self: Self, max_sample: int) -> None:
        """Update the maximum value of the progress bar.

        Parameters
        ----------
        max_sample : int
            The new maximum value of the progress bar.
        """
        self.pb_prediction.setMaximum(max_sample)

    def _update_sample_idx(self: Self, sample: int) -> None:
        """Update the value of the progress bar.

        Parameters
        ----------
        sample : int
            The new value of the progress bar.
        """
        self.pb_prediction.setValue(sample + 1)
        self.pb_prediction.setFormat(
            f"Sample {sample+1}/{self.pred_status.max_samples}"
        )

    def _predict_button_clicked(self: Self) -> None:
        """Run the prediction on the images."""
        if self.pred_status is not None:
            if (
                self.pred_status.state == PredictionState.IDLE
                or self.train_status.state == TrainingState.DONE
                or self.pred_status.state == PredictionState.CRASHED
            ):
                self.pred_status.state = PredictionState.PREDICTING
                self.predict_button.setEnabled(False)
                self.stop_button.setEnabled(True)

    def _stop_button_clicked(self: Self) -> None:
        """Stop the prediction."""
        if self.pred_status is not None:
            if self.pred_status.state == PredictionState.PREDICTING:
                self.pred_status.state = PredictionState.STOPPED
                self.stop_button.setEnabled(False)

    def _update_button_from_train(self: Self, state: TrainingState) -> None:
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

    def _update_button_from_pred(self: Self, state: PredictionState) -> None:
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
            self.stop_button.setEnabled(False)


if __name__ == "__main__":
    import sys

    from qtpy.QtWidgets import QApplication

    # Create a QApplication instance
    app = QApplication(sys.argv)

    # create signal
    train_status = TrainingStatus()  # type: ignore
    pred_status = PredictionStatus()  # type: ignore
    pred_signal = PredictionSignal()  # type: ignore
    train_signal = TrainingSignal()  # type: ignore

    # Instantiate widget
    widget = PredictionWidget(train_status, pred_status, train_signal, pred_signal)

    # Show the widget
    widget.show()

    # Run the application event loop
    sys.exit(app.exec_())
