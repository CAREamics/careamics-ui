"""Custom widgets used to build the plugins."""

__all__ = [
    "AdvancedConfigurationWindow",
    "AxesWidget",
    "CAREConfigurationWindow",
    "CAREamicsBanner",
    "ConfigurationWidget",
    "FolderWidget",
    "N2NConfigurationWindow",
    "N2VConfigurationWindow",
    "PowerOfTwoSpinBox",
    "PredictDataWidget",
    "PredictionWidget",
    "SavingWidget",
    "ScrollWidgetWrapper",
    "TBPlotWidget",
    "TrainDataWidget",
    "TrainProgressWidget",
    "TrainingWidget",
    "create_double_spinbox",
    "create_gpu_label",
    "create_int_spinbox",
    "create_progressbar",
    "layer_choice",
    "load_button",
]

from .advanced_config import AdvancedConfigurationWindow
from .axes_widget import AxesWidget
from .banner_widget import CAREamicsBanner
from .care_config_window import CAREConfigurationWindow
from .folder_widget import FolderWidget
from .gpu_widget import create_gpu_label
from .magicgui_widgets import layer_choice, load_button
from .n2n_config_window import N2NConfigurationWindow
from .n2v_config_window import N2VConfigurationWindow
from .predict_data_widget import PredictDataWidget
from .prediction_widget import PredictionWidget
from .qt_widgets import (
    PowerOfTwoSpinBox,
    create_double_spinbox,
    create_int_spinbox,
    create_progressbar,
)
from .saving_widget import SavingWidget
from .scroll_wrapper import ScrollWidgetWrapper
from .tbplot_widget import TBPlotWidget
from .train_data_widget import TrainDataWidget
from .train_progress_widget import TrainProgressWidget
from .training_configuration_widget import ConfigurationWidget
from .training_widget import TrainingWidget
