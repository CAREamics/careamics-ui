from careamics.utils import autocorrelation
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

try:
    import napari
    from napari.layers import Image as ImageLayer
except ImportError:
    _has_napari = False
else:
    _has_napari = True


class AutocorrelationWidget(QWidget):
    """Autocorrelation Widget."""

    def __init__(
        self,
        napari_viewer: napari.Viewer | None = None,
    ) -> None:
        """Initialize the widget.

        Parameters
        ----------
        napari_viewer : napari.Viewer or None, default=None
            Napari viewer.
        """
        super().__init__()
        self.viewer = napari_viewer
        self.autocorr_button = QPushButton("Calc. Autocorrelation")
        self.autocorr_button.clicked.connect(self._calc_autocorrelation)
        label = QLabel(
            """
            This method is used to explore spatial correlations in images,
            in particular in the noise.

            The autocorrelation is normalized to the zero-shift value,
            which is centered in the resulting images.
            """
        )
        vbox = QVBoxLayout()
        vbox.addWidget(label)
        vbox.addWidget(self.autocorr_button, alignment=Qt.AlignCenter)  # type: ignore
        self.setLayout(vbox)

    def _calc_autocorrelation(self):
        if self.viewer is not None:
            layer = self.viewer.layers.selection.active
            if layer is not None and isinstance(layer, ImageLayer):
                img = layer.data
                autocorr_img = autocorrelation(img)
                self.viewer.add_image(autocorr_img, name="Autocorrelation")
