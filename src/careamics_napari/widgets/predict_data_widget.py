"""A widget used to select a path or layer for prediction."""

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFormLayout,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from careamics_napari.widgets import FolderWidget, layer_choice

# if TYPE_CHECKING:
#     import napari
#     from napari.layers import Image

# at run time
try:
    import napari
except ImportError:
    _has_napari = False
else:
    _has_napari = True


class PredictDataWidget(QTabWidget):
    """A widget offering to select a layer from napari or a path from disk."""

    def __init__(self) -> None:
        """Initialize the widget."""
        super().__init__()

        # QTabs
        layer_tab = QWidget()
        disk_tab = QWidget()

        # add tabs
        _tab_idx = 0
        if _has_napari and napari.current_viewer() is not None:
            # tab for selecting data from napari layers
            self.addTab(layer_tab, "From layers")
            self.setTabToolTip(_tab_idx, "Use images from napari layers")
            # add tab contents
            self._set_layer_tab(layer_tab)
            _tab_idx += 1
        # tab for selecting data from disk
        self.addTab(disk_tab, "From disk")
        self.setTabToolTip(_tab_idx, "Use patches saved on the disk")
        self._set_disk_tab(disk_tab)

    def get_data_sources(self) -> str | np.ndarray | None:
        """Get the selected data sources."""
        if (
            self.img_pred.value is None  # type: ignore
            and len(self.pred_images_folder.get_folder()) == 0
        ):
            # no prediction data has been selected
            return None

        if self.currentIndex() == 0:
            # data is expected from napari layers
            pred_data = self.img_pred.value.data  # type: ignore
        else:
            # data is expected from disk
            pred_data = self.pred_images_folder.get_folder()

        return pred_data

    def _set_layer_tab(
        self,
        layer_tab: QWidget,
    ) -> None:
        """Set up the layer tab.

        Parameters
        ----------
        layer_tab : QWidget
            The layer tab.
        """
        form = QFormLayout()
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)  # type: ignore
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)  # type: ignore
        form.setContentsMargins(12, 12, 0, 0)

        self.img_pred = layer_choice()
        self.img_pred.native.setToolTip("Select the prediction layer.")
        form.addRow("Predict", self.img_pred.native)

        vbox = QVBoxLayout()
        vbox.addLayout(form)
        layer_tab.setLayout(vbox)

    def _set_disk_tab(self, disk_tab: QWidget) -> None:
        """Set up the disk tab.

        Parameters
        ----------
        disk_tab : QWidget
            The disk tab.
        """
        form = QFormLayout()
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)  # type: ignore
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)  # type: ignore
        form.setContentsMargins(12, 12, 0, 0)

        self.pred_images_folder = FolderWidget("Choose")
        self.pred_images_folder.setToolTip("Select a folder containing images.")
        form.addRow("Predict", self.pred_images_folder)

        vbox = QVBoxLayout()
        vbox.addLayout(form)
        disk_tab.setLayout(vbox)


if __name__ == "__main__":
    # from qtpy.QtWidgets import QApplication
    # import sys

    # # Create a QApplication instance
    # app = QApplication(sys.argv)

    # # create signal
    # signal = ConfigurationSignal()

    # # Instantiate widget
    # widget = DataSelectionWidget(signal, True)

    # # Show the widget
    # widget.show()

    # # Run the application event loop
    # sys.exit(app.exec_())

    import napari

    viewer = napari.Viewer()
    viewer.window.add_dock_widget(PredictDataWidget())

    napari.run()
