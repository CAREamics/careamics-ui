"""A widget displaying the training progress using two progress bars."""

from qtpy.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
)

from careamics_napari.careamics_utils import BaseConfig
from careamics_napari.signals import TrainingState, TrainingStatus
from careamics_napari.widgets import TBPlotWidget, create_progressbar


class TrainProgressWidget(QGroupBox):
    """A widget displaying the training progress using two progress bars.

    Parameters
    ----------
    careamics_config : Configuration
            careamics configuration object.
    train_status : TrainingStatus or None, default=None
        Signal representing the training status.
    """

    def __init__(
        self,
        careamics_config: BaseConfig,
        train_status: TrainingStatus | None = None,
    ) -> None:
        """Initialize the widget.

        Parameters
        ----------
        careamics_config : Configuration
            careamics configuration object.
        train_status : TrainingStatus or None, default=None
            Signal representing the training status.
        """
        super().__init__()

        self.configuration = careamics_config
        self.train_status = (
            train_status
            if train_status is not None  # for typing purposes
            else TrainingStatus()  # type: ignore
        )

        self.setTitle("Training Progress")
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 0)

        # progress bars
        self.pb_epochs = create_progressbar(
            max_value=self.train_status.max_epochs,
            text_format=f"Epoch ?/{self.train_status.max_epochs}",
            value=0,
        )

        self.pb_batch = create_progressbar(
            max_value=self.train_status.max_batches,
            text_format=f"Batch ?/{self.train_status.max_batches}",
            value=0,
        )

        # plot widget
        self.plot = TBPlotWidget(
            max_width=300,
            max_height=300,
            min_height=250,
            work_dir=self.configuration.work_dir,
        )

        layout.addWidget(self.pb_epochs)
        layout.addWidget(self.pb_batch)
        layout.addWidget(self.plot.native)
        self.setLayout(layout)

        # set actions based on the training status
        self.train_status.events.state.connect(self._update_training_state)
        self.train_status.events.epoch_idx.connect(self._update_epoch)
        self.train_status.events.max_epochs.connect(self._update_max_epoch)
        self.train_status.events.batch_idx.connect(self._update_batch)
        self.train_status.events.max_batches.connect(self._update_max_batch)
        self.train_status.events.val_loss.connect(self._update_loss)

    def _update_training_state(self, state: TrainingState) -> None:
        """Update the widget according to the training state.

        Parameters
        ----------
        state : TrainingState
            Training state.
        """
        if state == TrainingState.IDLE or state == TrainingState.TRAINING:
            self.plot.clear_plot()

    def _update_max_epoch(self, max_epoch: int):
        """Update the maximum number of epochs in the progress bar.

        Parameters
        ----------
        max_epoch : int
            Maximum number of epochs.
        """
        self.pb_epochs.setMaximum(max_epoch)

    def _update_epoch(self, epoch: int) -> None:
        """Update the epoch progress bar.

        Parameters
        ----------
        epoch : int
            Current epoch.
        """
        self.pb_epochs.setValue(epoch + 1)
        self.pb_epochs.setFormat(f"Epoch {epoch + 1}/{self.train_status.max_epochs}")

    def _update_max_batch(self, max_batches: int) -> None:
        """Update the maximum number of batches in the progress bar.

        Parameters
        ----------
        max_batches : int
            Maximum number of batches.
        """
        self.pb_batch.setMaximum(max_batches)

    def _update_batch(self) -> None:
        """Update the batch progress bar."""
        self.pb_batch.setValue(self.train_status.batch_idx + 1)
        self.pb_batch.setFormat(
            f"Batch {self.train_status.batch_idx + 1}/{self.train_status.max_batches}"
        )

    def _update_loss(self) -> None:
        """Update the loss plot."""
        self.plot.update_plot(
            epoch=self.train_status.epoch_idx,
            train_loss=self.train_status.loss,
            val_loss=self.train_status.val_loss,
        )


if __name__ == "__main__":
    # import sys
    import napari

    # from qtpy.QtWidgets import QApplication
    from careamics_napari.careamics_utils import get_default_n2v_config

    config = get_default_n2v_config()
    # app = QApplication(sys.argv)
    # widget = TrainProgressWidget(config)
    # widget.show()
    # sys.exit(app.exec_())
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(TrainProgressWidget(config))
    napari.run()
