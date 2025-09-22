"""
Training Signal base class.

Includes common parameters that will be updated by the UI.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from careamics.utils import get_careamics_home
from psygnal import evented

if TYPE_CHECKING:
    from psygnal import SignalGroup, SignalInstance

    # Class only declared at time checking in order to allow the autocompletion and
    # type checker to show the "events" attribute of the @evented class.
    class TrainingSignalGroup(SignalGroup):
        """Signal group for the training status dataclass."""

        # only parameters that have observers are listed here
        algorithm: SignalInstance
        """Algorithm used for training."""

        use_channels: SignalInstance
        """Whether the data has channels."""

        is_3d: SignalInstance
        """Whether the data is 3D."""


try:
    from napari.layers import Image
except ImportError:
    _has_napari = False
    """Whether napari is installed."""
else:
    _has_napari = True

HOME = get_careamics_home()


@evented
@dataclass
class TrainingSignal:
    """Training signal class.

    This class holds the parameters required to run the training thread. These
    parameters should be set whenever the user interact with the corresponding UI
    elements. An instance of the class is then passed to the training worker.
    """

    if TYPE_CHECKING:
        events: TrainingSignalGroup
        """Attribute allowing the registration of parameter-specific listeners."""

    # signals used to change states across widgets
    algorithm: str = "n2v"
    """Algorithm used for training."""

    use_channels: bool = False
    """Whether the data has channels."""

    is_3d: bool = False
    """Whether the data is 3D."""

    # parameters set by widgets for training
    work_dir: Path = HOME
    """Directory where the checkpoints and logs are saved."""

    load_from_disk: bool = True
    """Whether to load the images from disk or from the viewer."""

    if _has_napari:
        layer_train: Image | None = None
        """Layer containing the training data."""

        layer_train_target: Image | None = None
        """Layer containing the training target data."""

        layer_val: Image | None = None
        """Layer containing the validation data."""

        layer_val_target: Image | None = None
        """Layer containing the validation target data."""

    path_train: str = ""
    """Path to the training data."""

    path_train_target: str = ""
    """Path to the training target data."""

    path_val: str = ""
    """Path to the validation data."""

    path_val_target: str = ""
    """Path to the validation target."""

    axes: str = "YX"
    """Axes of the data."""

    patch_size_xy: int = 64
    """Size of the patches along the X and Y dimensions."""

    patch_size_z: int = 16
    """Size of the patches along the Z dimension."""

    n_epochs: int = 30
    """Number of epochs."""

    batch_size: int = 16
    """Batch size."""

    experiment_name = ""
    """Name of the experiment, used to export the model and save checkpoints."""

    x_flip: bool = True
    """Whether to apply flipping along the X dimension during augmentation."""

    y_flip: bool = True
    """Whether to apply flipping along the Y dimension during augmentation."""

    rotations: bool = True
    """Whether to apply rotations during augmentation."""

    independent_channels: bool = True
    """Whether to train the channels independently."""

    # n_channels_in_care: int = 1
    # """Number of input channels when training CARE and Noise2Noise."""

    # n_channels_out_care: int = 1
    # """Number of output channels when training CARE and Noise2Noise."""

    depth: int = 2
    """Depth of the U-Net."""

    num_conv_filters: int = 32
    """Number of convolutional filters in the first layer."""

    val_percentage: float = 0.1
    """Percentage of the training data used for validation."""

    val_minimum_split: int = 1
    """Minimum number of patches or images in the validation set."""
