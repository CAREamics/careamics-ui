"""Various pure Qt widgets."""

import math

from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QProgressBar,
    QSpinBox,
)
from typing_extensions import Any, Self


class SpinBox(QSpinBox):
    """A spin box that ignores wheel events."""

    def wheelEvent(self: Self, event: Any) -> None:
        """Ignore wheel events.

        Parameters
        ----------
        event : Any
            The wheel event.
        """
        event.ignore()


class DoubleSpinBox(QDoubleSpinBox):
    """A double spin box that ignores wheel events."""

    def wheelEvent(self: Self, event: Any) -> None:
        """Ignore wheel events.

        Parameters
        ----------
        event : Any
            The wheel event.
        """
        event.ignore()


class PowerOfTwoSpinBox(QSpinBox):
    """A spin box that only accepts power of two values.

    Parameters
    ----------
    min_val : int
        Minimum value.
    max_val : int
        Maximum value.
    default : int
        Default value.
    *args : Any
        Additional arguments.
    **kwargs : Any
        Additional keyword arguments.
    """

    def __init__(
        self: Self, min_val: int, max_val: int, default: int, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize the widget.

        Parameters
        ----------
        min_val : int
            Minimum value.
        max_val : int
            Maximum value.
        default : int
            Default value.
        *args : Any
            Additional arguments.
        **kwargs : Any
            Additional keyword arguments.

        Raises
        ------
        ValueError
            If min or max are not power of 2.
        """
        super().__init__(*args, **kwargs)

        # min or max are not power of 2
        if min_val & (min_val - 1) != 0:
            raise ValueError(f"Minimum value must be a power of 2, got {min_val}.")

        if max_val & (max_val - 1) != 0:
            raise ValueError(f"Maximum value must be a power of 2, got {max_val}.")

        self.setRange(min_val, max_val)
        self.setSingleStep(1)
        self.setValue(default)

    def stepBy(self: Self, steps: int) -> None:
        """Step the value by a given number of steps.

        Parameters
        ----------
        steps : int
            Number of steps to step the value by.
        """
        current_value = self.value()
        current_power = self._get_base_2_log(current_value)

        # Step up or down by adjusting the power of two
        new_power = current_power + steps
        new_value = 2**new_power
        self.setValue(new_value)

    def _get_base_2_log(self: Self, value: int) -> int:
        """Get base-2 logarithm.

        Parameters
        ----------
        value : int
            The value to get the power of two for.

        Returns
        -------
        int
            The power of two of the given value.
        """
        return int(math.log2(value))

    def textFromValue(self: Self, value: int) -> str:
        """Get the text representation of the value.

        Parameters
        ----------
        value : int
            The value.

        Returns
        -------
        str
            The text representation of the value.
        """
        return str(value)

    def valueFromText(self: Self, text: str) -> int:
        """Get the value from the text representation.

        Parameters
        ----------
        text : str
            The text representation.

        Returns
        -------
        int
            The value.
        """
        return int(text)


def create_double_spinbox(
    min_value: float = 0,
    max_value: float = 1,
    value: float = 0.5,
    step: float = 0.1,
    visible: bool = True,
    tooltip: str | None = None,
    n_decimal: int = 1,
) -> DoubleSpinBox:
    """Create a double-typed spin box.

    Parameters
    ----------
    min_value : float, default=0
        Minimum value.
    max_value : float, default=1
        Maximum value.
    value : float, default=0.5
        Default value.
    step : float, default=0.1
        Step value.
    visible : bool, default=True
        Visibility.
    tooltip : str or None, default=None
        Tooltip text.
    n_decimal : int, default=1
        Number of decimal places.

    Returns
    -------
    DoubleSpinBox
        The double spin box.
    """
    spin_box = DoubleSpinBox()
    spin_box.setDecimals(n_decimal)
    spin_box.setMinimum(min_value)
    spin_box.setMaximum(max_value)
    spin_box.setSingleStep(step)
    spin_box.setValue(value)
    spin_box.setVisible(visible)
    spin_box.setToolTip(tooltip)
    spin_box.setMinimumHeight(25)
    # spin_box.setContentsMargins(0, 3, 0, 3)
    return spin_box


def create_int_spinbox(
    min_value: int = 1,
    max_value: int = 1000,
    value: int = 2,
    step: int = 1,
    visible: bool = True,
    tooltip: str | None = None,
) -> SpinBox:
    """Create an integer-typed spin box.

    Parameters
    ----------
    min_value : int, default=1
        Minimum value.
    max_value : int, default=1000
        Maximum value.
    value : int, default=2
        Default value.
    step : int, default=1
        Step value.
    visible : bool, default=True
        Visibility.
    tooltip : str or None, default=None
        Tooltip text.

    Returns
    -------
    SpinBox
        The integer spin box.
    """
    spin_box = SpinBox()
    spin_box.setMinimum(min_value)
    spin_box.setMaximum(max_value)
    spin_box.setSingleStep(step)
    spin_box.setValue(value)
    spin_box.setVisible(visible)
    spin_box.setToolTip(tooltip)
    spin_box.setMinimumHeight(25)
    # spin_box.setContentsMargins(0, 3, 0, 3)

    return spin_box


def create_progressbar(
    min_value: int = 0,
    max_value: int = 100,
    value: int = 0,
    text_visible: bool = True,
    visible: bool = True,
    text_format: str = f"Epoch ?/{100}",
    tooltip: str | None = None,
) -> QProgressBar:
    """Create a progress bar.

    Parameters
    ----------
    min_value : int, default=0
        Minimum value.
    max_value : int, default=100
        Maximum value.
    value : int, default=0
        Default value.
    text_visible : bool, default=True
        Visibility of the text.
    visible : bool, default=True
        Visibility.
    text_format : str, default="Epoch ?/{100}"
        Text format.
    tooltip : str or None, default=None
        Tooltip text.

    Returns
    -------
    QProgressBar
        The progress bar.
    """
    progress_bar = QProgressBar()
    progress_bar.setMinimum(min_value)
    progress_bar.setMaximum(max_value)
    progress_bar.setValue(value)
    progress_bar.setVisible(visible)
    progress_bar.setTextVisible(text_visible)
    progress_bar.setFormat(text_format)
    progress_bar.setToolTip(tooltip)
    progress_bar.setMinimumHeight(30)

    return progress_bar
