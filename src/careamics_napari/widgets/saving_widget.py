"""A widget allowing users to select a model type and a path."""

from pathlib import Path

from qtpy.QtCore import Qt, Signal  # type: ignore
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QPushButton,
    QVBoxLayout,
)

from careamics_napari.careamics_utils import BaseConfig
from careamics_napari.signals import (
    ExportType,
    TrainingState,
    TrainingStatus,
)


class SavingWidget(QGroupBox):
    """A widget allowing users to export and save a model.

    Parameters
    ----------
    careamics_config : BaseConfig
        The configuration for the CAREamics algorithm.
    careamist : CAREamist
            Instance of CAREamist.
    train_status : TrainingStatus or None, default=None
        Signal containing training parameters.
    """

    export_model = Signal(Path, str)

    def __init__(
        self,
        careamics_config: BaseConfig,
        # careamist: CAREamist | None = None,
        train_status: TrainingStatus | None = None,
    ) -> None:
        """Initialize the widget.

        Parameters
        ----------
        careamics_config : BaseConfig
            The configuration for the CAREamics algorithm.
        careamist : CAREamist
            Instance of CAREamist.
        train_status : TrainingStatus or None, default=None
            Signal containing training parameters.
        """
        super().__init__()

        self.configuration = careamics_config
        # self.careamist = careamist
        self.train_status = train_status

        self.setTitle("Export")

        # format combobox
        self.save_choice = QComboBox()
        self.save_choice.addItems(ExportType.list())
        self.save_choice.setToolTip("Output format")

        self.save_button = QPushButton("Export Model")
        self.save_button.setMinimumWidth(120)
        self.save_button.setEnabled(False)
        self.save_button.setToolTip("Save the model weights and configuration.")

        # layout
        vbox = QVBoxLayout()
        vbox.addWidget(self.save_choice)
        vbox.addWidget(self.save_button, alignment=Qt.AlignLeft)  # type: ignore
        self.setLayout(vbox)

        # actions
        if self.train_status is not None:
            # updates from signals
            self.train_status.events.state.connect(self._update_training_state)
            # when clicking the save button
            self.save_button.clicked.connect(self._save_model)

    def _update_training_state(self, state: TrainingState) -> None:
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

    def _save_model(self) -> None:
        """Ask user for the destination folder and export the model."""
        destination = QFileDialog.getExistingDirectory(caption="Export Model")
        if len(destination) > 0:
            destination = Path(destination)
            export_type = self.save_choice.currentText()
            # emit export
            self.export_model.emit(destination, export_type)
