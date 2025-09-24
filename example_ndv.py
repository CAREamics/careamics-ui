# sanity checks for ndv example

try:
    import ndv
except ImportError:
    raise ImportError("You need to `pip install ndv` to run this example.")
from typing import cast

from qtpy import QT6

if not QT6:
    raise ImportError("ndv will require QT>=6.4.")

from careamics.utils import get_careamics_home
from careamics_portfolio import PortfolioManager
from napari.qt import get_stylesheet

from careamics_napari.training_plugin import TrainPlugin, TrainPluginWrapper
from careamics_napari.widgets import PredictDataWidget

# create a viewer and your main gui
viewer = ndv.ArrayViewer()
scroll = TrainPluginWrapper(viewer)
# apply the napari stylesheet
scroll.setStyleSheet(get_stylesheet("dark"))
scroll.show()

#  get sample data
files = PortfolioManager().denoising.N2V_SEM.download(path=get_careamics_home())

# my gross hack to pre-populate the text fields with the sample files :)
wdg = cast("TrainPlugin", scroll.widget())
wdg.data_layers[0].train_images_folder.text_field.setText(files[-2])
wdg.data_layers[0].val_images_folder.text_field.setText(files[-1])
pred_wdg = wdg.prediction_widget.findChild(PredictDataWidget)
pred_wdg.pred_images_folder.text_field.setText(files[-1])

# run the Qt event loop
ndv.run_app()
