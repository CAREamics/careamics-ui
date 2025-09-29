"""A widget allowing users to select a model type and a path."""

import traceback
from pathlib import Path

from qtpy.QtCore import Qt, Signal  # type: ignore
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QPushButton,
    QVBoxLayout,
)

# from careamics import CAREamist
from careamics_napari.careamics_utils import BaseConfig
from careamics_napari.signals import (
    ExportType,
    TrainingState,
    TrainingStatus,
)

try:
    import napari.utils.notifications as ntf
except ImportError:
    _has_napari = False
else:
    _has_napari = True


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
        destination = Path(QFileDialog.getExistingDirectory(caption="Export Model"))
        export_type = self.save_choice.currentText()
        # emit export
        # self._export_model(destination, export_type)
        self.export_model.emit(destination, export_type)

    def _export_model(self, destination: Path, export_type: str) -> None:
        dims = "3D" if self.configuration.is_3D else "2D"
        algo_name = self.configuration.algorithm_config.get_algorithm_friendly_name()
        name = f"{algo_name}_{dims}_{self.configuration.experiment_name}"

        try:
            if export_type == ExportType.BMZ:
                raise NotImplementedError("Export to BMZ not implemented yet (but soon).")
            elif self.careamist is not None:
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
