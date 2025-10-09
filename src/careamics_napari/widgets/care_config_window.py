from careamics.config.algorithms import CAREAlgorithm
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    # QCheckBox,
    QFormLayout,
    QVBoxLayout,
    QWidget,
)

from careamics_napari.careamics_utils import BaseConfig, CAREAdvancedConfig
from careamics_napari.widgets import AdvancedConfigurationWindow
from careamics_napari.widgets.qt_widgets import (
    # create_double_spinbox,
    create_int_spinbox,
)
from careamics_napari.widgets.utils import bind


class CAREConfigurationWindow(AdvancedConfigurationWindow):
    """A dialog widget for modifying CARE advanced settings."""

    def __init__(
        self,
        parent: QWidget | None,
        careamics_config: BaseConfig,
        algorithm_config: CAREAdvancedConfig,
    ) -> None:
        """Initialize the window.

        Parameters
        ----------
        parent : QWidget | None
            Parent widget.
        careamics_config : BaseConfig
            Careamics configuration object.
        algorithm_config : CAREAdvancedConfig
            CARE advanced configuration object.
        """
        super().__init__(parent, careamics_config, algorithm_config)

        self.advanced_configuration = algorithm_config

        self.add_algorithm_specific_tab()

        self.bind_properties()

    def add_algorithm_specific_tab(self) -> None:
        """Add algorithm specific advanced settings tab."""
        # tab widget
        tab_widget = QWidget()

        # number of input channels
        _n_channels = self.advanced_configuration.n_channels_in or 1
        self.num_channels_in_spin = create_int_spinbox(1, 10, _n_channels, 1)
        self.num_channels_in_spin.setEnabled(
            "C" in self.configuration.data_config.axes  # type: ignore
        )
        self.num_channels_in_spin.setToolTip(
            "Number of input channels of the input image (C must be in axes)."
        )

        # number of output channels
        _n_channels = self.advanced_configuration.n_channels_out or 1
        self.num_channels_out_spin = create_int_spinbox(1, 10, _n_channels, 1)
        # self.num_channels_out_spin.setEnabled(
        #     "C" in self.configuration.data_config.axes  # type: ignore
        # )
        self.num_channels_out_spin.setToolTip("Number of output channels.")

        # layout
        layout = QVBoxLayout()
        form = QFormLayout()
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)  # type: ignore
        form.setFieldGrowthPolicy(
            QFormLayout.AllNonFixedFieldsGrow  # type: ignore
        )
        form.addRow("Number of Input Channels:", self.num_channels_in_spin)
        form.addRow("Number of Output Channels:", self.num_channels_out_spin)
        layout.addLayout(form)

        tab_widget.setLayout(layout)
        self.tabs.addTab(tab_widget, "CARE")

    def save(self) -> None:
        """Save the current state of the UI into configurations."""
        super().update_config()
        self.update_config()
        self.close()

    def bind_properties(self) -> None:
        """Create and bind the properties to the UI elements."""
        # bind the properties from the base class first
        super().bind_properties()
        # type(self) returns the class of the instance, so we are adding
        # properties to the class itself, not the instance.
        type(self).in_channels = bind(self.num_channels_in_spin, "value")
        type(self).out_channels = bind(self.num_channels_out_spin, "value")

    def update_config(self) -> None:
        """Update the configuration object from UI elements."""
        self.advanced_configuration.n_channels_in = self.in_channels
        self.advanced_configuration.n_channels_out = self.out_channels

        if isinstance(self.configuration.algorithm_config, CAREAlgorithm):
            self.configuration.algorithm_config.model.in_channels = self.in_channels
            self.configuration.algorithm_config.model.num_classes = self.out_channels


if __name__ == "__main__":
    import sys

    from qtpy.QtWidgets import QApplication

    from careamics_napari.careamics_utils import get_default_n2v_config

    config = get_default_n2v_config()
    care_config = CAREAdvancedConfig()
    # Create a QApplication instance
    app = QApplication(sys.argv)
    widget = CAREConfigurationWindow(None, config, care_config)
    widget.show()

    sys.exit(app.exec_())
