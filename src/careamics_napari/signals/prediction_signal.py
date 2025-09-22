"""Prediction parameters set by the user."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from psygnal import evented

if TYPE_CHECKING:
    from napari.layers import Image

try:
    from napari.layers import Image
except ImportError:
    _has_napari = False
    """Whether napari is installed."""
else:
    _has_napari = True


# TODO should this class be evented? Probably not (it is not type checked currently)
@evented
@dataclass
class PredictionSignal:
    """Prediction signal class.

    This class holds the parameters required to run the prediction thread. These
    parameters should be set whenever the user interact with the corresponding UI
    elements. An instance of the class is then passed to the prediction worker.
    """

    load_from_disk: bool = True
    """Whether to load the images from disk or from the viewer."""

    if _has_napari:
        layer_pred: Image = None
        """Layer containing the data on which to predict."""

    path_pred: str = ""
    """Path to the data on which to predict."""

    is_3d: bool = False
    """Whether the data is 3D or 2D."""

    tiled: bool = False
    """Whether to predict the data in tiles."""

    tile_size_xy: int = 64
    """Size of the tiles along the X and Y dimensions."""

    tile_size_z: int = 8
    """Size of the tiles along the Z dimension."""

    tile_overlap_xy: int = 48  # TODO currently fixed
    """Overlap between the tiles along the X and Y dimensions."""

    tile_overlap_z: int = 4  # TODO currently fixed
    """Overlap between the tiles along the Z dimension."""
    
    batch_size: int = 1
    """Batch size."""
