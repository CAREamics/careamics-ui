"""A widget allowing users to select data source for the training."""

# from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFormLayout,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from careamics_napari.careamics_utils import BaseConfig
from careamics_napari.widgets import FolderWidget, layer_choice

# at run time
try:
    import napari
except ImportError:
    _has_napari = False
else:
    _has_napari = True


class TrainDataWidget(QTabWidget):
    """A widget offering to select layers from napari or paths from disk.

    Parameters
    ----------
    careamics_config : Configuration
            careamics configuration object.
    use_target : bool, default=False
        Whether to target fields.
    """

    def __init__(
        self,
        careamics_config: BaseConfig,
        use_target: bool = False,
    ) -> None:
        """Initialize the widget.

        Parameters
        ----------
        careamics_config : Configuration
            careamics configuration object.
        use_target : bool, default=False
            Whether to target fields.
        """
        super().__init__()
        self.configuration = careamics_config
        self.use_target = use_target

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

    def get_data_sources(self) -> dict[str, list] | None:
        """Get the selected data sources."""
        if (
            self.img_train.value is None  # type: ignore
            and len(self.train_images_folder.get_folder()) == 0
        ):
            # no training data has been selected
            return None

        if self.currentIndex() == 0:
            # data is expected from napari layers
            train_data = [self.img_train.value.data]  # type: ignore
            val_data = [self.img_val.value.data]  # type: ignore
            if self.use_target:
                train_data.append(self.target_train.value.data)  # type: ignore
                val_data.append(self.target_val.value.data)  # type: ignore

        else:
            # data is expected from disk
            train_data = [self.train_images_folder.get_folder()]
            val_data = [self.val_images_folder.get_folder()]
            if self.use_target:
                train_data.append(self.train_target_folder.get_folder())
                val_data.append(self.val_target_folder.get_folder())

        return {"train": train_data, "val": val_data}

    def _set_layer_tab(self, layer_tab: QWidget) -> None:
        """Set up the layer tab.

        Parameters
        ----------
        layer_tab : QWidget
            Layer tab widget.
        """
        form = QFormLayout()
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)  # type: ignore
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)  # type: ignore
        form.setContentsMargins(12, 12, 0, 0)

        self.img_train = layer_choice()
        self.img_train.native.setToolTip("Select the training layer.")

        self.img_val = layer_choice()
        self.img_train.native.setToolTip("Select the validation layer.")

        form.addRow("Train", self.img_train.native)
        form.addRow("Val", self.img_val.native)

        if self.use_target:
            # get the target layers
            self.target_train = layer_choice()
            self.target_val = layer_choice()

            # tool tips
            self.target_train.native.setToolTip("Select a training target layer.")
            self.target_val.native.setToolTip("Select a validation target layer.")

            form.addRow("Train target", self.target_train.native)
            form.addRow("Val target", self.target_val.native)

        vbox = QVBoxLayout()
        vbox.addLayout(form)
        layer_tab.setLayout(vbox)

    def _set_disk_tab(self, disk_tab: QWidget) -> None:
        """Set up the disk tab.

        Parameters
        ----------
        disk_tab : QWidget
            Disk tab widget.
        """
        form = QFormLayout()
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)  # type: ignore
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)  # type: ignore
        form.setContentsMargins(12, 12, 0, 0)

        self.train_images_folder = FolderWidget("Choose")
        self.val_images_folder = FolderWidget("Choose")
        form.addRow("Train", self.train_images_folder)
        form.addRow("Val", self.val_images_folder)

        if self.use_target:
            self.train_target_folder = FolderWidget("Choose")
            self.val_target_folder = FolderWidget("Choose")

            form.addRow("Train target", self.train_target_folder)
            form.addRow("Val target", self.val_target_folder)

            self.train_target_folder.setToolTip(
                "Select a folder containing the training\ntarget."
            )
            self.val_target_folder.setToolTip(
                "Select a folder containing the validation\ntarget."
            )
            self.train_images_folder.setToolTip(
                "Select a folder containing the training\nimages."
            )
            self.val_images_folder.setToolTip(
                "Select a folder containing the validation\nimages."
            )

        else:
            self.train_images_folder.setToolTip(
                "Select a folder containing the training\nimages."
            )
            self.val_images_folder.setToolTip(
                "Select a folder containing the validation\n"
                "images, if you select the same folder as\n"
                "for training, the validation patches will\n"
                "be extracted from the training data."
            )

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

    # import qdarktheme
    import napari

    from careamics_napari.careamics_utils import get_default_n2v_config

    # qdarktheme.enable_hi_dpi()

    config = get_default_n2v_config()
    # create a Viewer
    viewer = napari.Viewer()
    # add napari-n2v plugin
    viewer.window.add_dock_widget(TrainDataWidget(config, True))

    # add image to napari
    # viewer.add_image(data[0][0], name=data[0][1]['name'])
    # start UI
    # qdarktheme.setup_theme("auto")
    napari.run()
