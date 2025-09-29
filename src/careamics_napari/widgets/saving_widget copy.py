"""A widget allowing users to select a model type and a path."""

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from typing_extensions import Self

from careamics_napari.signals import (
    ExportType,
    SavingSignal,
    SavingState,
    SavingStatus,
    TrainingState,
    TrainingStatus,
)


class SavingWidget(QGroupBox):
    """A widget allowing users to select a model type and a path.

    Parameters
    ----------
    train_status : TrainingStatus or None, default=None
        Signal containing training parameters.
    save_status : SavingStatus or None, default=None
        Signal containing saving parameters.
    save_signal : SavingSignal or None, default=None
        Signal to trigger saving.
    """

    def __init__(
        self: Self,
        train_status: TrainingStatus | None = None,
        save_status: SavingStatus | None = None,
        save_signal: SavingSignal | None = None,
    ) -> None:
        """Initialize the widget.

        Parameters
        ----------
        train_status : TrainingStatus or None, default=None
            Signal containing training parameters.
        save_status : SavingStatus or None, default=None
            Signal containing saving parameters.
        save_signal : SavingSignal or None, default=None
            Signal to trigger saving.
        """
        super().__init__()

        self.train_status = train_status
        self.save_status = save_status
        self.save_signal = save_signal

        self.setTitle("Save")
        self.setLayout(QVBoxLayout())

        # Save button
        save_widget = QWidget()
        save_widget.setLayout(QHBoxLayout())
        self.save_choice = QComboBox()
        self.save_choice.addItems(ExportType.list())
        self.save_choice.setToolTip("Output format")

        self.save_button = QPushButton("Save model", self)
        self.save_button.setMinimumWidth(120)
        self.save_button.setEnabled(False)
        self.save_choice.setToolTip("Save the model weights and configuration.")

        save_widget.layout().addWidget(self.save_choice)
        save_widget.layout().addWidget(self.save_button, alignment=Qt.AlignLeft)
        self.layout().addWidget(save_widget)

        # actions
        if self.train_status is not None:
            # updates from signals
            self.train_status.events.state.connect(self._update_training_state)

            # when clicking the save button
            self.save_button.clicked.connect(self._save_model)

            # when changing the format
            self.save_choice.currentIndexChanged.connect(self._update_export_type)

    def _update_export_type(self: Self, index: int) -> None:
        """Set the signal export type to the selected format.

        Parameters
        ----------
        index : int
            Index of the selected format.
        """
        if self.save_signal is not None:
            self.save_signal.export_type = self.save_choice.currentText()

    def _update_training_state(self: Self, state: TrainingState) -> None:
        """Update the widget state based on the training state.

        Parameters
        ----------
        state : TrainingState
            Current training state.
        """
        if state == TrainingState.DONE or state == TrainingState.STOPPED:
            self.save_button.setEnabled(True)
        elif state == TrainingState.IDLE:
            self.save_button.setEnabled(False)

    def _save_model(self: Self) -> None:
        """Prompt users with a path selection dialog and update the saving state."""
        if self.save_status is not None:
            if self.save_signal is not None and (
                self.save_status.state == SavingState.IDLE
                or self.save_status.state == SavingState.DONE
                or self.save_status.state == SavingState.CRASHED
            ):
                # destination = Path(QFileDialog.getSaveFileName(caption='Save model'))
                destination = Path(QFileDialog.getExistingDirectory(caption="Save model"))
                self.save_signal.path_model = destination

                # trigger saving
                self.save_status.state = SavingState.SAVING


if __name__ == "__main__":
    import sys

    from qtpy.QtWidgets import QApplication

    # Create a QApplication instance
    app = QApplication(sys.argv)

    # create signal
    train_signal = TrainingStatus()  # type: ignore
    save_signal = SavingSignal()  # type: ignore
    save_status = SavingStatus()  # type: ignore

    # Instantiate widget
    widget = SavingWidget(train_signal, save_status, save_signal)

    # Show the widget
    widget.show()

    # Run the application event loop
    sys.exit(app.exec_())
