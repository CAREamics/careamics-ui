from careamics.config import create_n2n_configuration

from careamics_napari.utils import get_num_workers

from .configs import AdvancedConfig, BaseConfig


class N2NAdvancedConfig(AdvancedConfig):
    """N2N advanced configuration."""

    n_channels_in: int = 1
    """Number of input channels."""

    n_channels_out: int = 1
    """Number of output channels."""


def get_default_n2n_config() -> BaseConfig:
    """Return a default N2N configuration."""
    num_workers = get_num_workers()

    config = create_n2n_configuration(
        experiment_name="careamics_care",
        data_type="array",
        axes="YX",
        patch_size=[64, 64],
        batch_size=16,
        num_epochs=30,
        independent_channels=True,
        train_dataloader_params={"num_workers": num_workers},
        val_dataloader_params={"num_workers": num_workers},
    )
    config = BaseConfig(**config.model_dump(), needs_gt=True)

    return config
