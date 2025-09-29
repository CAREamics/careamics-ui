"""Saving parameters set by the user."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ExportType(Enum):
    """Type of model export."""

    CKPT = "Checkpoint"
    """PyTorch Lightning checkpoint."""

    BMZ = "Bioimage.io"
    """Bioimage.io model format."""

    @classmethod
    def list(cls) -> list[str]:
        """List of all available export types.

        Returns
        -------
        list of str
            List of all available export types.
        """
        return [c.value for c in cls]


@dataclass
class SavingSignal:
    """Saving signal class.

    This class holds the parameters required to run the prediction thread. These
    parameters should be set whenever the user interact with the corresponding UI
    elements.
    """

    path_model: Path = Path(".")
    """Path in which to save the model."""

    export_type: ExportType = ExportType.BMZ
    """Format of model export."""
