from pathlib import Path

from careamics.config import Configuration, create_n2v_configuration
from careamics.utils import get_careamics_home

from careamics_napari.utils import get_num_workers

HOME = get_careamics_home()


class BaseConfig(Configuration):
    """Base configuration class."""

    needs_gt: bool = False
    """Whether the algorithm requires ground truth (for training)."""

    use_channels: bool = False
    """Whether the data has channels."""

    is_3D: bool = False
    """Whether the data is 3D."""

    work_dir: Path = HOME
    """Directory where the checkpoints and logs are saved."""

    val_percentage: float = 0.1
    """Percentage of the training data used for validation."""

    val_minimum_split: int = 1
    """Minimum number of patches or images in the validation set."""

    # prediction parameters
    tile_size: tuple[int, int] | None = None
    """Size of the tiles to predict on."""

    tile_overlap_xy: int = 48
    """Overlap between the tiles along the X and Y dimensions."""

    tile_overlap_z: int = 4
    """Overlap between the tiles along the Z dimension."""

    pred_batch_size: int = 1
    """Batch size for prediction."""


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
        train_dataloader_params={"num_workers": num_workers},
        val_dataloader_params={"num_workers": num_workers},
    )
    config = BaseConfig(**config.model_dump(), needs_gt=False)

    return config


if __name__ == "__main__":
    config = get_default_n2v_config()
    print(config)
