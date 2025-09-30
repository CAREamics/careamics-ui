from careamics.config.algorithms import N2VAlgorithm
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QVBoxLayout,
    QWidget,
)

from careamics_napari.careamics_utils import BaseConfig, N2VAdvancedConfig
from careamics_napari.widgets import AdvancedConfigurationWindow
from careamics_napari.widgets.qt_widgets import (
    create_double_spinbox,
    create_int_spinbox,
)
from careamics_napari.widgets.utils import bind


class N2VConfigurationWindow(AdvancedConfigurationWindow):
    """A dialog widget for modifying N2V advanced settings."""

    def __init__(
        self,
        parent: QWidget | None,
        careamics_config: BaseConfig,
        algorithm_config: N2VAdvancedConfig,
    ) -> None:
        """Initialize the window.

        Parameters
        ----------
        parent : QWidget | None
            Parent widget.
        careamics_config : BaseConfig
            Careamics configuration object.
        algorithm_config : N2VAdvancedConfig
            N2V advanced configuration object.
        """
        super().__init__(parent, careamics_config, algorithm_config)

        self.advanced_configuration = algorithm_config

        self.add_algorithm_specific_tab()

        self.bind_properties()

    def add_algorithm_specific_tab(self) -> None:
        """Add algorithm specific advanced settings tab."""
        # tab widget
        tab_widget = QWidget()

        # use n2v2 checkbox
        self.n2v2_chkbox = QCheckBox("Use N2V2")
        self.n2v2_chkbox.setChecked(self.advanced_configuration.use_n2v2)
        self.n2v2_chkbox.setToolTip("If checked, will use N2V2 instead of N2V")

        # roi size spin box
        self.roi_spin = create_int_spinbox(
            1, 101, self.advanced_configuration.roi_size, 2
        )
        self.roi_spin.setToolTip(
            "The size of the area around each pixel "
            "that will be manipulated by algorithm (must be an odd number)."
        )

        # masked pixel percentage double spin box
        self.masked_percentage_spin = create_double_spinbox(
            0.0,
            1.0,
            self.advanced_configuration.masked_pixel_percentage,
            0.01,
            n_decimal=2,
        )
        self.masked_percentage_spin.setToolTip(
            "Percentage of pixels that per patch that will be manipulated."
        )

        # number of channels
        _n_channels = self.advanced_configuration.n_channels or 1
        self.num_channels_spin = create_int_spinbox(1, 10, _n_channels, 1)
        self.num_channels_spin.setEnabled(
            "C" in self.configuration.data_config.axes  # type: ignore
        )
        self.num_channels_spin.setToolTip(
            "Number of channels in the input image (C must be in axes)."
        )

        # layout
        layout = QVBoxLayout()
        layout.addWidget(self.n2v2_chkbox)
        layout.addSpacing(15)
        form = QFormLayout()
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)  # type: ignore
        form.setFieldGrowthPolicy(
            QFormLayout.AllNonFixedFieldsGrow  # type: ignore
        )
        form.addRow("ROI Size:", self.roi_spin)
        form.addRow("Masked Pixel Percentage:", self.masked_percentage_spin)
        form.addRow("Number of Channels:", self.num_channels_spin)
        layout.addLayout(form)

        tab_widget.setLayout(layout)
        self.tabs.addTab(tab_widget, "N2V")

    def save(self) -> None:
        """Save the current state of the UI into configurations."""
        super().update_config()
        self.update_config()
        self.close()

    def bind_properties(self) -> None:
        """Create and bind the properties to the UI elements."""
        super().bind_properties()
        type(self).use_n2v2 = bind(self.n2v2_chkbox, "checked")
        type(self).roi_size = bind(self.roi_spin, "value")
        type(self).masked_pixel_percentage = bind(self.masked_percentage_spin, "value")
        type(self).n_channels = bind(self.num_channels_spin, "value")

    def update_config(self) -> None:
        """Update the configuration object from UI elements."""
        self.advanced_configuration.use_n2v2 = self.use_n2v2
        self.configuration.algorithm_config.set_n2v2(self.use_n2v2)  # type: ignore

        self.advanced_configuration.roi_size = self.roi_size
        self.advanced_configuration.masked_pixel_percentage = self.masked_pixel_percentage
        self.advanced_configuration.n_channels = self.n_channels

        if isinstance(self.configuration.algorithm_config, N2VAlgorithm):
            self.configuration.algorithm_config.n2v_config.roi_size = self.roi_size
            self.configuration.algorithm_config.n2v_config.masked_pixel_percentage = (
                self.masked_pixel_percentage
            )
            self.configuration.algorithm_config.model.in_channels = self.n_channels
            self.configuration.algorithm_config.model.num_classes = self.n_channels


if __name__ == "__main__":
    import sys

    from qtpy.QtWidgets import QApplication

    from careamics_napari.careamics_utils import get_default_n2v_config

    config = get_default_n2v_config()
    n2v_config = N2VAdvancedConfig()
    # Create a QApplication instance
    app = QApplication(sys.argv)
    widget = N2VConfigurationWindow(None, config, n2v_config)
    widget.show()

    sys.exit(app.exec_())
