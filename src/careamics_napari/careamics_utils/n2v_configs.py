from careamics.config import create_n2v_configuration

from careamics_napari.utils import get_num_workers

from .configs import AdvancedConfig, BaseConfig


class N2VAdvancedConfig(AdvancedConfig):
    """N2V advanced configuration."""

    use_n2v2: bool = False
    """To use N2V2"""

    roi_size: int = 11
    """The size of the area around each pixel that will be manipulated by N2V."""

    masked_pixel_percentage: float = 0.2
    """How many pixels per patch will be manipulated."""

    n_channels: int | None = None
    """Number of channels in the input image (C must be in axes)."""

    # struct_n2v_axis = None
    # struct_n2v_span = 5


def get_default_n2v_config() -> BaseConfig:
    """Return default N2V configuration."""
    num_workers = get_num_workers()

    config = create_n2v_configuration(
        experiment_name="careamics",
        data_type="array",
        axes="YX",
        patch_size=[64, 64],
        batch_size=16,
        num_epochs=30,
        independent_channels=True,
        train_dataloader_params={"num_workers": num_workers},
        val_dataloader_params={"num_workers": num_workers},
    )
    config = BaseConfig(**config.model_dump(), needs_gt=False)

    return config
