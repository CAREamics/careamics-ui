"""Widget for specifying axes order."""

from enum import Enum
from typing import Any

from careamics.config.architectures import UNetModel
from qtpy import QtGui
from qtpy.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QWidget
from typing_extensions import Self

from careamics_napari.careamics_utils import BaseConfig
from careamics_napari.utils import REF_AXES, are_axes_valid
from careamics_napari.widgets.utils import bind


class Highlight(Enum):
    """Axes highlight types."""

    VALID = 0
    """Valid axes."""

    UNRECOGNIZED = 1
    """Unrecognized axes."""

    NOT_ACCEPTED = 2
    """Axes not accepted."""


class LettersValidator(QtGui.QValidator):
    """Custom validator.

    Parameters
    ----------
    options : str
        Allowed characters.
    *args : Any
        Variable length argument list.
    **kwargs : Any
        Arbitrary keyword arguments.
    """

    def __init__(self: Self, options: str, *args: Any, **kwargs: Any) -> None:
        """Initialize the validator.

        Parameters
        ----------
        options : str
            Allowed characters.
        *args : Any
            Variable length argument list.
        **kwargs : Any
            Arbitrary keyword arguments.
        """
        QtGui.QValidator.__init__(self, *args, **kwargs)
        self._options = options

    def validate(
        self: Self, value: str, pos: int
    ) -> tuple[QtGui.QValidator.State, str, int]:
        """Validate the input.

        Parameters
        ----------
        value : str
            Input value.
        pos : int
            Position of the cursor.

        Returns
        -------
        (QtGui.QValidator.State, str, int)
            Validation state, value, and position.
        """
        if len(value) > 0:
            if value[-1] in self._options:
                return QtGui.QValidator.Acceptable, value, pos  # type: ignore
        else:
            if value == "":
                return QtGui.QValidator.Intermediate, value, pos  # type: ignore
        return QtGui.QValidator.Invalid, value, pos  # type: ignore


# TODO keep the validation?
# TODO is train layer selected, then show the orange and red, otherwise ignore?
class AxesWidget(QWidget):
    """A widget allowing users to specify axes.

    Parameters
    ----------
        careamics_config : Configuration
            careamics configuration object.
    """

    def __init__(
        self,
        careamics_config: BaseConfig,
    ) -> None:
        """Initialize the widget.

        Parameters
        ----------
            training_signal : BaseConfig
                A careamics configuration object.
        """
        super().__init__()
        self.configuration = careamics_config
        self.is_text_valid = True

        # layout
        layout = QHBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # text field
        self.label = QLabel("Axes")
        self.text_field = QLineEdit(careamics_config.data_config.axes)  # type: ignore
        self.text_field.setMaxLength(6)
        self.text_field.setValidator(LettersValidator(REF_AXES))
        self.text_field.textChanged.connect(self._validate_axes)  # type: ignore
        self.text_field.setToolTip(
            "Enter the axes order as they are in your images, e.g. SZYX.\n"
            "Accepted axes are S(ample), T(ime), C(hannel), Z, Y, and X. Red\n"
            "color highlighting means that a character is not recognized,\n"
            "orange means that the axes order is not allowed. YX axes are\n"
            "mandatory."
        )
        # layout
        layout.addWidget(self.label)
        layout.addWidget(self.text_field)
        self.setLayout(layout)
        # validate text
        self._validate_axes(self.text_field.text())

        # create and bind properties to ui
        type(self).axes = bind(
            self.text_field,
            "text",
            default_value=self.configuration.data_config.axes,  # type: ignore
            validation_fn=self._validate_axes,
        )

    def update_config(self: Self) -> None:
        """Update the axes in the configuration if it's valid."""
        if self.is_text_valid:
            self.configuration.data_config.axes = self.axes  # type: ignore
            if isinstance(self.configuration.algorithm_config.model, UNetModel):
                self.configuration.algorithm_config.model.independent_channels = (
                    "C" in self.axes
                )
            # print(self.configuration)

    def _validate_axes(self: Self, axes: str | None = None) -> bool:
        """Validate the input text in the text field."""
        if axes is None:
            axes = self.text_field.text()
        # change text color according to axes validation
        if are_axes_valid(axes):
            self._set_text_color(Highlight.VALID)
            self.is_text_valid = True
            # if axes.upper() in filter_dimensions(self.n_axes, self.is_3D):
            #     self._set_text_color(Highlight.VALID)
            # else:
            #     self._set_text_color(Highlight.NOT_ACCEPTED)
        else:
            self._set_text_color(Highlight.UNRECOGNIZED)
            self.is_text_valid = False

        return self.is_text_valid

    def _set_text_color(self: Self, highlight: Highlight) -> None:
        """Set the text color according to the highlight type.

        Parameters
        ----------
        highlight : Highlight
            Highlight type.
        """
        if highlight == Highlight.UNRECOGNIZED:
            self.text_field.setStyleSheet("color: red;")
        elif highlight == Highlight.NOT_ACCEPTED:
            self.text_field.setStyleSheet("color: orange;")
        else:  # VALID
            self.text_field.setStyleSheet("color: white;")


if __name__ == "__main__":
    import sys

    from qtpy.QtWidgets import QApplication, QPushButton

    from careamics_napari.careamics_utils import get_default_n2v_config

    config = get_default_n2v_config()
    # Create a QApplication instance
    app = QApplication(sys.argv)
    widget = AxesWidget(careamics_config=config)
    btn = QPushButton("test")
    btn.clicked.connect(widget.update_config)
    widget.layout().addWidget(btn)  # type: ignore
    # Show the widget
    widget.show()

    # Run the application event loop
    sys.exit(app.exec_())
