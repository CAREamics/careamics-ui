"""N2V plugin."""

from careamics_napari.base_plugin import BasePlugin
from careamics_napari.careamics_utils import N2NAdvancedConfig, get_default_n2n_config
from careamics_napari.widgets import N2NConfigurationWindow

try:
    import napari
except ImportError:
    _has_napari = False
else:
    _has_napari = True


class N2NPlugin(BasePlugin):
    """CAREamics N2N plugin.

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

        # create a n2n config
        self.careamics_config = get_default_n2n_config()
        # advanced n2n config
        self.advanced_config = N2NAdvancedConfig()

        # assemble plugin ui
        self.add_careamics_banner(
            "CAREamics UI for training Noise2Noise (N2N) denoising models."
        )
        self.add_train_input_ui(use_target=self.careamics_config.needs_gt)
        self.add_config_ui()
        self.add_train_button_ui()
        self.add_prediction_ui()
        self.add_model_export_ui()

    def show_advanced_config(self) -> None:
        """Show advanced configuration."""
        self.update_config()
        # show window with advanced options
        win = N2NConfigurationWindow(self, self.careamics_config, self.advanced_config)
        win.finished.connect(lambda: print(self.advanced_config, self.careamics_config))
        win.show()


if __name__ == "__main__":
    import napari

    # create a Viewer
    viewer = napari.Viewer()
    # add n2v plugin
    viewer.window.add_dock_widget(N2NPlugin(viewer))
    # start UI
    napari.run()
