"""N2V Training parameters that will be updated by the UI."""

from dataclasses import dataclass
from typing import Literal

from psygnal import evented

from careamics_napari.signals.training_signal import TrainingSignal


@evented
@dataclass
class N2VTrainingSignal(TrainingSignal):
    """N2V Training signal class."""

    n_channels_n2v: int = 1
    """Number of channels when training Noise2Void."""

    use_n2v2: bool = False
    """Whether to use N2V2."""

    roi_size: int = 11
    """N2V pixel manipulation area."""

    masked_pixel_percentage: float = 0.2
    """Percentage of pixels masked in each patch."""

    struct_n2v_axis: Literal["horizontal", "vertical", "none"] = "none"
    """Axis along which to apply structN2V mask."""

    struct_n2v_span: int = 5
    """Span of the structN2V mask."""
