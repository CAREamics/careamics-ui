"""Magicgui widgets."""

from pathlib import Path
from typing import Any, Optional

from magicgui import magic_factory
from magicgui.widgets import Widget, create_widget

# at run time
try:
    from napari import current_viewer
    from napari.layers import Image
except ImportError:
    _has_napari = False
else:
    _has_napari = True


def layer_choice(annotation: Optional[Any] = Image, **kwargs: Any) -> Widget:
    """Create a widget to select a layer from the napari viewer.

    Parameters
    ----------
    annotation : Any or None, default=Image
        The annotation type to filter the layers.
    **kwargs : Any
        Additional keyword arguments to pass to the widget.

    Returns
    -------
    Widget
        The widget to select a layer from the napari viewer.

    Raises
    ------
    ImportError
        If napari is not installed.
    """
    if not _has_napari:
        raise ImportError("napari is not installed.")

    widget: Widget = create_widget(annotation=annotation, **kwargs)
    widget.reset_choices()  # type: ignore
    viewer = current_viewer()

    # connect to napari events
    if viewer is not None:
        viewer.layers.events.inserted.connect(widget.reset_choices)  # type: ignore
        viewer.layers.events.removed.connect(widget.reset_choices)  # type: ignore
        viewer.layers.events.changed.connect(widget.reset_choices)  # type: ignore

    return widget


@magic_factory(auto_call=True, Model={"mode": "r", "filter": "*.ckpt *.zip"})
def load_button(Model: Path):
    """A button to load model files.

    Parameters
    ----------
    Model : pathlib.Path
        The path to the model file.
    """
    pass
