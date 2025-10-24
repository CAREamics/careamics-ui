"""A widget allowing the creation of a CAREamics configuration."""

from careamics.config.data import DataConfig
from qtpy import QtGui
from qtpy.QtCore import Qt, Signal  # type: ignore
from qtpy.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QVBoxLayout,
)

from careamics_napari.careamics_utils import BaseConfig
from careamics_napari.resources import ICON_GEAR
from careamics_napari.widgets import (
    AxesWidget,
    PowerOfTwoSpinBox,
    create_int_spinbox,
)
from careamics_napari.widgets.utils import bind


class ConfigurationWidget(QGroupBox):
    """A widget allowing the creation of a CAREamics configuration.

    Parameters
    ----------
    careamics_config : Configuration
        careamics configuration object.
    """

    # signal to show algorithm advanced configuration window.
    show_advanced_config = Signal()

    def __init__(self, careamics_config: BaseConfig) -> None:
        """Initialize the widget.

        Parameters
        ----------
        careamics_config : Configuration
            careamics configuration object.
        """
        super().__init__()

        self.configuration = careamics_config

        self.setTitle("Training Parameters")
        self.setMinimumWidth(200)

        # advanced settings
        icon = QtGui.QIcon(ICON_GEAR)
        self.training_expert_btn = QPushButton(icon, "")
        self.training_expert_btn.setFixedSize(35, 35)
        self.training_expert_btn.setToolTip("Open the advanced settings window.")
        self.training_expert_btn.clicked.connect(lambda: self.show_advanced_config.emit())

        # 3D checkbox
        self.enable_3d_chkbox = QCheckBox()
        self.enable_3d_chkbox.setToolTip("Use a 3D network")
        self.enable_3d_chkbox.clicked.connect(self._enable_3d_changed)

        # axes
        self.axes_widget = AxesWidget(careamics_config=self.configuration)

        # number of epochs
        _n_epochs = 30
        if self.configuration.training_config.lightning_trainer_config is not None:
            _n_epochs = self.configuration.training_config.lightning_trainer_config[
                "max_epochs"
            ]
        self.n_epochs_spin = create_int_spinbox(
            1, 1000, _n_epochs, tooltip="Number of epochs"
        )

        # batch size
        self.batch_size_spin = create_int_spinbox(1, 512, 16, 1)
        self.batch_size_spin.setToolTip(
            "Number of patches per batch (decrease if GPU memory is insufficient)"
        )

        # patch size XY
        self.patch_xy_spin = PowerOfTwoSpinBox(16, 512, 64)
        self.patch_xy_spin.setToolTip("Dimension of the patches in XY.")
        # patch size Z
        self.patch_z_spin = PowerOfTwoSpinBox(8, 512, 8)
        self.patch_z_spin.setToolTip("Dimension of the patches in Z.")
        self.patch_z_spin.setEnabled(self.configuration.is_3D)

        # layout
        formLayout = QFormLayout()
        formLayout.setContentsMargins(0, 0, 0, 0)
        formLayout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)  # type: ignore
        formLayout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)  # type: ignore
        formLayout.addRow("Enable 3D", self.enable_3d_chkbox)
        formLayout.addRow(self.axes_widget.label.text(), self.axes_widget.text_field)
        formLayout.addRow("# Epochs", self.n_epochs_spin)
        formLayout.addRow("Batch size", self.batch_size_spin)
        formLayout.addRow("Patch XY", self.patch_xy_spin)
        formLayout.addRow("Patch Z", self.patch_z_spin)
        formLayout.minimumSize()

        vbox = QVBoxLayout()
        vbox.setContentsMargins(5, 20, 5, 10)
        vbox.addWidget(
            self.training_expert_btn,
            alignment=Qt.AlignRight | Qt.AlignVCenter,  # type: ignore
        )
        vbox.addLayout(formLayout)
        self.setLayout(vbox)

        # create and bind properties to ui
        self._bind_properties()

    def update_config(self) -> None:
        """Update the configuration from the UI element."""
        # num epochs
        if self.configuration.training_config.lightning_trainer_config is not None:
            self.configuration.training_config.lightning_trainer_config["max_epochs"] = (
                self.num_epochs
            )

        if isinstance(self.configuration.data_config, DataConfig):
            # batch size
            self.configuration.data_config.batch_size = self.batch_size
            # is 3D
            self.configuration.is_3D = self.is_3D
            # patch size
            _patch_size = [self.patch_xy_size, self.patch_xy_size]
            if self.is_3D:
                _patch_size.insert(0, self.patch_z_size)
                # update the configuration
            self.configuration.set_3D(self.is_3D, self.axes_widget.axes, _patch_size)

    def _enable_3d_changed(self, state: bool) -> None:
        """Update the signal 3D state.

        Parameters
        ----------
        state : bool
            3D state.
        """
        self.patch_z_spin.setEnabled(state)
        self.configuration.is_3D = self.is_3D
        self.axes_widget.validate_axes()

    def _bind_properties(self) -> None:
        """Create and bind the properties to the UI elements."""
        # type(self) returns the class of the instance, so we are adding
        # properties to the class itself, not the instance.
        # is 3D
        type(self).is_3D = bind(self.enable_3d_chkbox, "checked")
        # number of epochs
        if self.configuration.training_config.lightning_trainer_config is not None:
            type(self).num_epochs = bind(self.n_epochs_spin, "value")

        if isinstance(self.configuration.data_config, DataConfig):
            # batch size
            type(self).batch_size = bind(self.batch_size_spin, "value")
            # XY patch size
            type(self).patch_xy_size = bind(self.patch_xy_spin, "value")
            # Z patch size
            type(self).patch_z_size = bind(self.patch_z_spin, "value")


if __name__ == "__main__":
    import sys

    from qtpy.QtWidgets import QApplication

    from careamics_napari.careamics_utils import get_default_n2v_config

    config = get_default_n2v_config()
    # Create a QApplication instance
    app = QApplication(sys.argv)
    widget = ConfigurationWidget(config)
    widget.show()

    # Run the application event loop
    sys.exit(app.exec_())
