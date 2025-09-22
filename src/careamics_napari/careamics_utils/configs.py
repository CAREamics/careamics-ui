from pathlib import Path

from careamics.utils import get_careamics_home
from careamics.config import (
    Configuration,
    create_n2v_configuration
)


HOME = get_careamics_home()


class BaseConfig(Configuration):
    """Base configuration class."""

    use_channels: bool = False
    """Whether the data has channels."""

    is_3D: bool = False
    """Whether the data is 3D."""

    work_dir: Path = HOME
    """Directory where the checkpoints and logs are saved."""


def get_default_n2v_config() -> BaseConfig:
    """Return default N2V configuration."""
    config = create_n2v_configuration(
        experiment_name="careamics",
        data_type="array",
        axes="YX",
        patch_size=[64, 64],
        batch_size=16,
        num_epochs=30
    )
    config = BaseConfig(
        **config.model_dump(),
    )

    return config


if __name__ == "__main__":
    config = get_default_n2v_config()
    print(config)
