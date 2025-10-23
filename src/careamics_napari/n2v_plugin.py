"""N2V plugin."""

from careamics_napari.base_plugin import BasePlugin
from careamics_napari.careamics_utils import N2VAdvancedConfig, get_default_n2v_config
from careamics_napari.widgets import N2VConfigurationWindow

try:
    import napari
except ImportError:
    _has_napari = False
else:
    _has_napari = True


class N2VPlugin(BasePlugin):
    """CAREamics N2V plugin.

    Parameters
    ----------
    napari_viewer : napari.Viewer or None, default=None
        Napari viewer.
    """

    def __init__(
        self,
        napari_viewer: napari.Viewer | None = None,
    ) -> None:
        """Initialize the plugin.

        Parameters
        ----------
        napari_viewer : napari.Viewer or None, default=None
            Napari viewer.
        """
        super().__init__(napari_viewer)
        self.viewer = napari_viewer

        # create a n2v config
        self.careamics_config = get_default_n2v_config()
        # advanced n2v config
        self.advanced_config = N2VAdvancedConfig()

        # assemble plugin ui
        self.add_careamics_banner(
            "CAREamics UI for training Noise2Void (N2V) denoising models."
        )
        self.add_train_input_ui(use_target=self.careamics_config.needs_gt)
        self.add_config_ui()
        self.add_train_button_ui()
        self.add_prediction_ui()
        self.add_model_export_ui()

    def show_advanced_config(self) -> None:
        """Show advanced configuration."""
        # show window with advanced options
        win = N2VConfigurationWindow(self, self.careamics_config, self.advanced_config)
        win.finished.connect(lambda: print(self.advanced_config, self.careamics_config))
        win.show()


if __name__ == "__main__":
    import napari

    # create a Viewer
    viewer = napari.Viewer()
    # add n2v plugin
    viewer.window.add_dock_widget(N2VPlugin(viewer))
    # start UI
    napari.run()
