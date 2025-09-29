from collections.abc import Callable
from typing import Any

from qtpy.QtWidgets import QWidget


def bind(
    widget: QWidget,
    prop_name: str,
    default_value: Any | None = None,
    validation_fn: Callable | None = None,
) -> property:
    """Returns a property binding to the given widget property.

    Parameters
    ----------
        widget: QWidget
            The widget whose property we want to bind to
        prop_name: str
            The name of the property in the widget (e.g. text)
        default_value: Any (optional)
            The default value to be used when the widget value is not valid.
            Defaults to None.
        validation_fn: Callable (optional)
        The validation function (must return a boolean). Defaults to None.

    Returns
    -------
        property: A property
    """

    def getter(self):
        ui_value = widget.property(prop_name)
        if validation_fn:
            if not validation_fn(ui_value):
                print(f"Invalid input: {ui_value}")
                ui_value = default_value

        return ui_value

    def setter(self, value):
        widget.setProperty(prop_name, value)

    return property(fget=getter, fset=setter)


if __name__ == "__main__":
    import sys

    from qtpy.QtWidgets import QApplication, QLineEdit, QPushButton, QVBoxLayout

    from careamics_napari.careamics_utils import get_default_n2v_config
    from careamics_napari.utils import are_axes_valid

    config = get_default_n2v_config()

    class Win(QWidget):
        """docstring for Win""" ""

        def __init__(self):
            super().__init__()
            line_edit = QLineEdit()
            # line_edit.setText(config.data_config.axes)
            btn = QPushButton("Test")
            btn.clicked.connect(lambda: print(self.axes))
            vbox = QVBoxLayout()
            vbox.addWidget(line_edit)
            vbox.addWidget(btn)
            self.setLayout(vbox)

            type(self).axes = bind(
                line_edit,
                "text",
                default_value=config.data_config.axes,  # type: ignore
                validation_fn=are_axes_valid,
            )
            self.axes = config.data_config.axes  # type: ignore

    app = QApplication([])
    win = Win()
    win.show()
    sys.exit(app.exec_())
