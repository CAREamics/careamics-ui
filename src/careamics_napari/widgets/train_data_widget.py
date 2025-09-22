"""A widget allowing users to select data source for the training."""

from typing import TYPE_CHECKING, Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFormLayout,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from typing_extensions import Self

from careamics_napari.signals import TrainingSignal
from careamics_napari.widgets import FolderWidget, layer_choice
from careamics_napari.widgets.utils import bind
from careamics_napari.careamics_utils import BaseConfig


if TYPE_CHECKING:
    import napari
    from napari.layers import Image

# at run time
try:
    import napari
    from napari.layers import Image
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
        self: Self,
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
        # layer_tab.setLayout(QVBoxLayout())
        disk_tab = QWidget()
        # disk_tab.setLayout(QVBoxLayout())

        # add tabs
        self.addTab(layer_tab, "From layers")
        self.addTab(disk_tab, "From disk")
        self.setTabToolTip(0, "Use images from napari layers")
        self.setTabToolTip(1, "Use patches saved on the disk")

        # set tabs
        self._set_layer_tab(layer_tab)
        self._set_disk_tab(disk_tab)

        # set active tab actions
        # self.currentChanged.connect(self._set_data_source)
        # self._set_data_source(self.currentIndex())

    def _set_layer_tab(
        self: Self,
        layer_tab: QWidget,
    ) -> None:
        """Set up the layer tab.

        Parameters
        ----------
        layer_tab : QWidget
            Layer tab widget.
        """
        if _has_napari and napari.current_viewer() is not None:
            form = QFormLayout()
            form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)  # type: ignore
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)  # type: ignore
            form.setContentsMargins(12, 12, 0, 0)

            self.img_train = layer_choice()
            self.img_train.native.setToolTip("Select the training layer.")

            self.img_val = layer_choice()
            self.img_train.native.setToolTip("Select the validation layer.")

            # connection actions for images
            # self.img_train.changed.connect(self._update_train_layer)
            # self.img_val.changed.connect(self._update_val_layer)

            # to cover the case when image was loaded before the plugin
            # if self.img_train.value is not None:
            #     self._update_train_layer(self.img_train.value)
            # if self.img_val.value is not None:
            #     self._update_val_layer(self.img_val.value)

            if self.use_target:
                # get the target layers
                self.target_train = layer_choice()
                self.target_val = layer_choice()

                # tool tips
                self.target_train.native.setToolTip("Select a training target layer.")
                self.target_val.native.setToolTip("Select a validation target layer.")

                # connection actions for targets
                # self.target_train.changed.connect(self._update_train_target_layer)
                # self.target_val.changed.connect(self._update_val_target_layer)

                # to cover the case when image was loaded before the plugin
                # if self.target_train.value is not None:
                #     self._update_train_target_layer(self.target_train.value)
                # if self.target_val.value is not None:
                #     self._update_val_target_layer(self.target_val.value)

                form.addRow("Train", self.img_train.native)
                form.addRow("Val", self.img_val.native)
                form.addRow("Train target", self.target_train.native)
                form.addRow("Val target", self.target_val.native)

            else:
                form.addRow("Train", self.img_train.native)
                form.addRow("Val", self.img_val.native)

            # layer_tab.layout().addLayout(form)
            vbox = QVBoxLayout()
            vbox.addLayout(form)
            layer_tab.setLayout(vbox)

        else:
            # simply remove the tab
            self.removeTab(0)

    def _set_disk_tab(self: Self, disk_tab: QWidget) -> None:
        """Set up the disk tab.

        Parameters
        ----------
        disk_tab : QWidget
            Disk tab widget.
        """
        # disk tab
        # buttons = QWidget()
        form = QFormLayout()
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)  # type: ignore
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)  # type: ignore
        # form.setSpacing(0)

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
                "Select a folder containing the training\n" "target."
            )
            self.val_target_folder.setToolTip(
                "Select a folder containing the validation\n" "target."
            )
            self.train_images_folder.setToolTip(
                "Select a folder containing the training\n" "images."
            )
            self.val_images_folder.setToolTip(
                "Select a folder containing the validation\n" "images."
            )

            # add actions to target
            # self.train_target_folder.get_text_widget().textChanged.connect(
            #     self._update_train_target_folder
            # )
            # self.val_target_folder.get_text_widget().textChanged.connect(
            #     self._update_val_target_folder
            # )

        else:
            self.train_images_folder.setToolTip(
                "Select a folder containing the training\n" "images."
            )
            self.val_images_folder.setToolTip(
                "Select a folder containing the validation\n"
                "images, if you select the same folder as\n"
                "for training, the validation patches will\n"
                "be extracted from the training data."
            )

        # add actions
        # self.train_images_folder.get_text_widget().textChanged.connect(
        #     self._update_train_folder
        # )
        # self.val_images_folder.get_text_widget().textChanged.connect(
        #     self._update_val_folder
        # )

        # buttons.setLayout(form)
        # disk_tab.layout().addWidget(buttons)
        vbox = QVBoxLayout()
        vbox.addLayout(form)
        disk_tab.setLayout(vbox)

    def _set_data_source(self: Self, index: int) -> None:
        """Set the signal data source to the selected tab.

        Parameters
        ----------
        index : int
            Index of the selected tab.
        """
        if self.configuration is not None:
            self.configuration.load_from_disk = index == self.count() - 1

    def _update_train_layer(self: Self, layer: Image) -> None:
        """Update the training layer.

        Parameters
        ----------
        layer : Image
            Training layer.
        """
        if self.configuration is not None:
            self.configuration.layer_train = layer

    def _update_val_layer(self: Self, layer: Image) -> None:
        """Update the validation layer.

        Parameters
        ----------
        layer : Image
            Validation layer.
        """
        if self.configuration is not None:
            self.configuration.layer_val = layer

    def _update_train_target_layer(self: Self, layer: Image) -> None:
        """Update the training target layer.

        Parameters
        ----------
        layer : Image
            Training target layer.
        """
        if self.configuration is not None:
            self.configuration.layer_train_target = layer

    def _update_val_target_layer(self: Self, layer: Image) -> None:
        """Update the validation target layer.

        Parameters
        ----------
        layer : Image
            Validation target layer.
        """
        if self.configuration is not None:
            self.configuration.layer_val_target = layer

    def _update_train_folder(self: Self, folder: str) -> None:
        """Update the training folder.

        Parameters
        ----------
        folder : str
            Training folder.
        """
        if self.configuration is not None:
            self.configuration.path_train = folder

    def _update_val_folder(self: Self, folder: str) -> None:
        """Update the validation folder.

        Parameters
        ----------
        folder : str
            Validation folder.
        """
        if self.configuration is not None:
            self.configuration.path_val = folder

    def _update_train_target_folder(self: Self, folder: str) -> None:
        """Update the training target folder.

        Parameters
        ----------
        folder : str
            Training target folder.
        """
        if self.configuration is not None:
            self.configuration.path_train_target = folder

    def _update_val_target_folder(self: Self, folder: str) -> None:
        """Update the validation target folder.

        Parameters
        ----------
        folder : str
            Validation target folder.
        """
        if self.configuration is not None:
            self.configuration.path_val_target = folder


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

    import qdarktheme
    import napari
    from careamics_napari.careamics_utils import get_default_n2v_config

    qdarktheme.enable_hi_dpi()

    config = get_default_n2v_config()
    # create a Viewer
    viewer = napari.Viewer()
    # add napari-n2v plugin
    viewer.window.add_dock_widget(
        TrainDataWidget(config, True)
    )

    # add image to napari
    # viewer.add_image(data[0][0], name=data[0][1]['name'])
    # start UI
    qdarktheme.setup_theme("auto")
    napari.run()
