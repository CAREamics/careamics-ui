"""Training widget."""

from typing import Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from typing_extensions import Self

from careamics_napari.signals import TrainingState, TrainingStatus


class TrainingWidget(QGroupBox):
    """Training widget.

    Parameters
    ----------
    train_status : TrainingStatus or None, default=None
        Training status.
    """

    def __init__(self: Self, train_status: Optional[TrainingStatus] = None) -> None:
        """Initialize the widget.

        Parameters
        ----------
        train_status : TrainingStatus or None, default=None
            Training status.
        """
        super().__init__()  # TODO needed? and in the other classes? to pass parent?

        self.train_status = (
            TrainingStatus() if train_status is None else train_status  # type: ignore
        )

        self.setTitle("Train")
        self.setLayout(QVBoxLayout())

        # train buttons
        train_buttons = QWidget()
        train_buttons.setLayout(QHBoxLayout())

        self.train_button = QPushButton("Train", self)
        self.train_button.setMinimumWidth(120)

        self.reset_model_button = QPushButton("Reset", self)
        self.reset_model_button.setMinimumWidth(120)
        self.reset_model_button.setEnabled(False)
        self.reset_model_button.setToolTip(
            "Reset the weights of the model (forget the training)"
        )

        train_buttons.layout().addWidget(self.train_button, alignment=Qt.AlignLeft)
        train_buttons.layout().addWidget(self.reset_model_button, alignment=Qt.AlignLeft)
        self.layout().addWidget(train_buttons)

        # actions
        if self.train_status is not None:
            # what to do when the buttons are clicked
            self.train_button.clicked.connect(self._train_stop_clicked)
            self.reset_model_button.clicked.connect(self._reset_clicked)

            # listening to the signal
            self.train_status.events.state.connect(self._update_button)

    def _train_stop_clicked(self) -> None:
        """Update the UI and training status when the train button is clicked."""
        if self.train_status is not None:
            if (
                self.train_status.state == TrainingState.IDLE
                or self.train_status.state == TrainingState.DONE
            ):
                self.train_status.state = TrainingState.TRAINING
                self.reset_model_button.setEnabled(False)
                # self.reset_model_button.setText("Reset")
                self.train_button.setText("Stop")

            elif self.train_status.state == TrainingState.TRAINING:
                self.train_status.state = TrainingState.STOPPED
                self.train_button.setText("Train")
                self.reset_model_button.setEnabled(True)
                # self.reset_model_button.setText("Reset")

            elif self.train_status.state == TrainingState.STOPPED:
                self.train_status.state = TrainingState.TRAINING
                self.train_button.setText("Stop")

    def _reset_clicked(self) -> None:
        """Update the UI and training status when the reset button is clicked."""
        if self.train_status is not None:
            if self.train_status.state != TrainingState.TRAINING:
                self.train_status.state = TrainingState.IDLE
                self.train_button.setText("Train")
                self.reset_model_button.setEnabled(False)

    def _update_button(self, new_state: TrainingState) -> None:
        """Update the button text based on the training state.

        Parameters
        ----------
        new_state : TrainingState
            New training state.
        """
        if new_state == TrainingState.DONE or new_state == TrainingState.STOPPED:
            self.train_button.setText("Train")
            self.reset_model_button.setEnabled(True)
            self.reset_model_button.setText("Reset")
        elif new_state == TrainingState.CRASHED:
            self._reset_clicked()


if __name__ == "__main__":
    import sys

    from qtpy.QtWidgets import QApplication

    # Create a QApplication instance
    app = QApplication(sys.argv)

    # create signal
    signal = TrainingStatus()  # type: ignore

    # Instantiate widget
    widget = TrainingWidget(signal)

    # Show the widget
    widget.show()

    # Run the application event loop
    sys.exit(app.exec_())
