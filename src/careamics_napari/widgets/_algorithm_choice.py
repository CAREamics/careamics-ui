"""Algorithm selection widget."""

from typing import Optional

from qtpy.QtWidgets import QComboBox

from careamics_napari.careamics_utils import get_algorithm, get_available_algorithms
from careamics_napari.signals import TrainingSignal


class AlgorithmSelectionWidget(QComboBox):
    """Algorithm selection widget.

    Parameters
    ----------
    training_signal : TrainingSignal or None, default=None
        Training signal holding all parameters to be set by the user.
    """

    def __init__(self, training_signal: Optional[TrainingSignal] = None) -> None:
        """Initialize the widget.

        Parameters
        ----------
        training_signal : TrainingSignal or None, default=None
            Training signal holding all parameters to be set by the user.
        """
        super().__init__()

        self.signal = training_signal

        self.addItems(get_available_algorithms())
        self.setToolTip("Select an algorithm.")

        # Connect the signal
        self.currentIndexChanged.connect(self.algorithm_changed)
        self.current_algorithm = self._get_current_algorithm()

    def _get_current_algorithm(self) -> str:
        """Return the current algorithm name.

        Returns
        -------
        str
            Current algorithm name.
        """
        return get_algorithm(self.currentText())

    def algorithm_changed(self, index: int) -> None:
        """Emit the algorithm signal.

        Parameters
        ----------
        index : int
            Index of the selected algorithm.
        """
        # save SupportedAlgorithm
        self.current_algorithm = self._get_current_algorithm()

        # emit the signal
        if self.signal is not None:
            self.signal.algorithm = self.current_algorithm


if __name__ == "__main__":
    import sys

    from qtpy.QtWidgets import QApplication

    myalgo = TrainingSignal()  # type: ignore

    @myalgo.events.algorithm.connect
    def print_algorithm(name: str):
        """Print the selected algorithm."""
        print(f"Selected algorithm: {name}")

    # Create a QApplication instance
    app = QApplication(sys.argv)

    # Instantiate widget
    widget = AlgorithmSelectionWidget(myalgo)

    # Show the widget
    widget.show()

    # Run the application event loop
    sys.exit(app.exec_())
