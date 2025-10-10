from pathlib import Path
from typing import Annotated

from careamics.config import Configuration
from careamics.utils import get_careamics_home
from pydantic import BaseModel, Field

HOME = get_careamics_home()


class BaseConfig(Configuration):
    """Base configuration class."""

    needs_gt: Annotated[bool, Field(exclude=True)] = False
    """Whether the algorithm requires ground truth (for training)."""

    use_channels: Annotated[bool, Field(exclude=True)] = False
    """Whether the data has channels."""

    is_3D: Annotated[bool, Field(exclude=True)] = False
    """Whether the data is 3D."""

    work_dir: Annotated[Path, Field(exclude=True)] = HOME
    """Directory where the checkpoints and logs are saved."""

    # training parameters
    val_percentage: Annotated[float, Field(exclude=True)] = 0.1
    """Percentage of the training data used for validation."""

    val_minimum_split: Annotated[int, Field(exclude=True)] = 1
    """Minimum number of patches or images in the validation set."""

    # prediction parameters
    tile_size: Annotated[
        tuple[int, int] | tuple[int, int, int] | None, Field(exclude=True)
    ] = None
    """Size of the tiles to predict on."""

    tile_overlap_xy: Annotated[int, Field(exclude=True)] = 48
    """Overlap between the tiles along the X and Y dimensions."""

    tile_overlap_z: Annotated[int, Field(exclude=True)] = 4
    """Overlap between the tiles along the Z dimension."""

    pred_batch_size: Annotated[int, Field(exclude=True)] = 1
    """Batch size for prediction."""


class AdvancedConfig(BaseModel):
    """Advanced configuration class."""

    x_flip: bool = True
    """Whether to apply flipping along the X dimension during augmentation."""

    y_flip: bool = True
    """Whether to apply flipping along the Y dimension during augmentation."""

    rotations: bool = True
    """Whether to apply rotations during augmentation."""
