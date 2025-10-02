"""A dialog widget allowing modifying advanced settings."""

from typing import Union

from careamics.config.architectures import UNetModel
from careamics.config.transformations import XYFlipModel, XYRandomRotate90Model
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from careamics_napari.careamics_utils import AdvancedConfig, BaseConfig
from careamics_napari.widgets.qt_widgets import (
    create_double_spinbox,
    create_int_spinbox,
)
from careamics_napari.widgets.utils import bind


class AdvancedConfigurationWindow(QDialog):
    """A dialog widget allowing modifying advanced settings.

    Parameters
    ----------
    parent : QWidget
        Parent widget.
    careamics_config : BaseConfig
            The configuration for the CAREamics algorithm.
    """

    def __init__(
        self,
        parent: QWidget | None,
        careamics_config: BaseConfig,
        algorithm_config: AdvancedConfig | None = None,
    ) -> None:
        """Initialize the widget.

        Parameters
        ----------
        parent : QWidget
            Parent widget.
        careamics_config : BaseConfig
            The configuration for the CAREamics algorithm.
        """
        super().__init__(parent)

        self.configuration = careamics_config
        self.advanced_configuration = algorithm_config

        self.tabs = QTabWidget()
        self.add_common_tab()

        self.save_button = QPushButton("Save")
        self.save_button.setMaximumWidth(120)
        self.save_button.clicked.connect(self.save)

        vbox = QVBoxLayout()
        vbox.addWidget(self.tabs)
        vbox.addWidget(self.save_button)
        self.setLayout(vbox)

        # self.bind_properties()

    def add_common_tab(self) -> None:
        """Add common advanced setting tab."""
        # tab widget
        tab_widget = QWidget()

        # experiment name text box
        label = QLabel("Experiment Name:")
        self.experiment_txtbox = QLineEdit()
        self.experiment_txtbox.setText(self.configuration.experiment_name)
        self.experiment_txtbox.setToolTip(
            "Name of the experiment. It will be used to save the model\n"
            "and the training history."
        )

        # validation
        val_grpbox = QGroupBox("Validation")
        self.val_perc_spin = create_double_spinbox(
            0.01, 1, self.configuration.val_percentage, 0.01, n_decimal=2
        )
        self.val_perc_spin.setToolTip(
            "Percentage of the training data used for validation."
        )
        self.val_split_spin = create_int_spinbox(
            1, 100, self.configuration.val_minimum_split, 1
        )
        self.val_split_spin.setToolTip(
            "Minimum number of patches or images in the validation set."
        )

        # augmentations group box, with x_flip, y_flip and rotations
        _x_flip = True
        _y_flip = True
        _rotations = True
        if self.advanced_configuration is not None:
            _x_flip = self.advanced_configuration.x_flip
            _y_flip = self.advanced_configuration.y_flip
            _rotations = self.advanced_configuration.rotations

        augment_grpbox = QGroupBox("Augmentations")
        self.x_flip_chkbox = QCheckBox("X Flip")
        self.x_flip_chkbox.setChecked(_x_flip)
        self.x_flip_chkbox.setToolTip(
            "Check to add augmentation that flips the image\nalong the x-axis"
        )
        self.y_flip_chkbox = QCheckBox("Y Flip")
        self.y_flip_chkbox.setChecked(_y_flip)
        self.y_flip_chkbox.setToolTip(
            "Check to add augmentation that flips the image\nalong the y-axis"
        )
        self.rotations_chkbox = QCheckBox("90 Rotations")
        self.rotations_chkbox.setChecked(_rotations)
        self.rotations_chkbox.setToolTip(
            "Check to add augmentation that rotates the image\n"
            "in 90 degree increments in XY"
        )

        # model params
        _depth = 2
        _num_filters = 32
        _indi_channels = True
        if isinstance(self.configuration.algorithm_config.model, UNetModel):
            _depth = self.configuration.algorithm_config.model.depth
            _num_filters = self.configuration.algorithm_config.model.num_channels_init
            _indi_channels = (
                self.configuration.algorithm_config.model.independent_channels
            )

        model_grpbox = QGroupBox("UNet parameters")
        self.model_depth_spin = create_int_spinbox(2, 5, _depth, 1)
        self.model_depth_spin.setToolTip("Depth of the U-Net model.")
        self.num_conv_filters_spin = create_int_spinbox(8, 1024, _num_filters, 8)
        self.num_conv_filters_spin.setToolTip(
            "Number of convolutional filters in the first layer."
        )

        # independent channels checkbox
        self.indi_channels_chkbox = QCheckBox("Independent Channels")
        self.indi_channels_chkbox.setChecked(_indi_channels)
        self.indi_channels_chkbox.setToolTip(
            "Check to treat the channels independently during\ntraining."
        )
        self.indi_channels_chkbox.setEnabled(
            "C" in self.configuration.data_config.axes  # type: ignore
        )

        # layout
        layout = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(label)
        hbox.addWidget(self.experiment_txtbox)
        layout.addLayout(hbox)
        layout.addSpacing(10)

        validation_layout = QFormLayout()
        validation_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)  # type: ignore
        validation_layout.setFieldGrowthPolicy(
            QFormLayout.AllNonFixedFieldsGrow  # type: ignore
        )
        validation_layout.addRow("Percentage", self.val_perc_spin)
        validation_layout.addRow("Minimum split", self.val_split_spin)
        val_grpbox.setLayout(validation_layout)
        layout.addWidget(val_grpbox)
        layout.addSpacing(10)

        augmentations_layout = QFormLayout()
        augmentations_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)  # type: ignore
        augmentations_layout.setFieldGrowthPolicy(
            QFormLayout.AllNonFixedFieldsGrow  # type: ignore
        )
        augmentations_layout.addRow(self.x_flip_chkbox)
        augmentations_layout.addRow(self.y_flip_chkbox)
        augmentations_layout.addRow(self.rotations_chkbox)
        augment_grpbox.setLayout(augmentations_layout)
        layout.addWidget(augment_grpbox)
        layout.addSpacing(10)

        model_params_layout = QFormLayout()
        model_params_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)  # type: ignore
        model_params_layout.setFieldGrowthPolicy(
            QFormLayout.AllNonFixedFieldsGrow  # type: ignore
        )
        model_params_layout.addRow("Depth", self.model_depth_spin)
        model_params_layout.addRow("# Filters", self.num_conv_filters_spin)
        model_grpbox.setLayout(model_params_layout)
        layout.addWidget(model_grpbox)
        layout.addSpacing(10)

        layout.addWidget(self.indi_channels_chkbox)

        tab_widget.setLayout(layout)
        self.tabs.addTab(tab_widget, "Common")

    def add_algorithm_specific_tab(self) -> None:
        """Add algorithm specific advanced settings tab."""
        raise NotImplementedError("Should be implemented by subclasses.")

    def save(self) -> None:
        """Save and close the window."""
        self.update_config()
        # self.close()  # should be closed in a subclass

    def bind_properties(self) -> None:
        """Create and bind the properties to the UI elements."""
        type(self).experiment_name = bind(self.experiment_txtbox, "text")
        type(self).val_percentage = bind(self.val_perc_spin, "value")
        type(self).val_split = bind(self.val_split_spin, "value")
        type(self).x_flip = bind(self.x_flip_chkbox, "checked")
        type(self).y_flip = bind(self.y_flip_chkbox, "checked")
        type(self).rotation = bind(self.rotations_chkbox, "checked")
        type(self).model_depth = bind(self.model_depth_spin, "value")
        type(self).num_conv_filters = bind(self.num_conv_filters_spin, "value")
        type(self).indi_channels = bind(self.indi_channels_chkbox, "checked")

    def update_config(self) -> None:
        """Update the configuration object from UI elements."""
        self.configuration.experiment_name = self.experiment_name

        self.configuration.val_percentage = self.val_percentage
        self.configuration.val_minimum_split = self.val_split

        if isinstance(self.configuration.algorithm_config.model, UNetModel):
            self.configuration.algorithm_config.model.depth = self.model_depth
            self.configuration.algorithm_config.model.num_channels_init = (
                self.num_conv_filters
            )

            self.configuration.algorithm_config.model.independent_channels = (
                self.indi_channels
            )

        # update augmentations
        augs: list[Union[XYFlipModel, XYRandomRotate90Model]] = []
        if self.x_flip or self.y_flip:
            augs.append(XYFlipModel(flip_x=self.x_flip, flip_y=self.y_flip, p=0.5))
        if self.rotation:
            augs.append(XYRandomRotate90Model(p=0.5))
        self.configuration.data_config.transforms = augs  # type: ignore
        # update advanced config as well
        if self.advanced_configuration is not None:
            self.advanced_configuration.x_flip = self.x_flip
            self.advanced_configuration.y_flip = self.y_flip
            self.advanced_configuration.rotations = self.rotation


if __name__ == "__main__":
    import sys

    from qtpy.QtWidgets import QApplication

    from careamics_napari.careamics_utils import get_default_n2v_config

    config = get_default_n2v_config()
    adv_config = AdvancedConfig()
    # Create a QApplication instance
    app = QApplication(sys.argv)
    widget = AdvancedConfigurationWindow(None, config, adv_config)
    widget.show()

    sys.exit(app.exec_())
