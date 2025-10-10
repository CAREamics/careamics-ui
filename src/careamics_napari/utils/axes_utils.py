"""Utilities to check axes validity."""

import warnings
from itertools import permutations

import numpy as np
from numpy.typing import NDArray

REF_AXES = "STCZYX"
"""References axes in CAREamics."""

NAPARI_AXES = "TSZYXC"  # TODO is this still true?
"""Axes used in Napari."""


def filter_dimensions(shape_length: int, is_3D: bool) -> list[str]:
    """Filter axes based on shape and dimensions.

    Parameters
    ----------
    shape_length : int
        Number of dimensions.
    is_3D : bool
        Whether the dimensions include Z.

    Returns
    -------
    list of str
        List of valid axes.
    """
    axes = list(REF_AXES)
    n = shape_length

    if not is_3D:  # if not 3D, remove it from the
        axes.remove("Z")

    if n > len(axes):
        warnings.warn("Data shape length is too large.", stacklevel=3)
        return []
    else:
        all_permutations = ["".join(p) for p in permutations(axes, n)]

        # X and Y must be in each permutation and contiguous (#FancyComments)
        all_permutations = [p for p in all_permutations if ("XY" in p) or ("YX" in p)]

        if is_3D:
            all_permutations = [p for p in all_permutations if "Z" in p]

        if len(all_permutations) == 0 and not is_3D:
            all_permutations = ["YX"]

        return all_permutations


# TODO should use function from CAREamics?
def are_axes_valid(axes: str) -> bool:
    """Check if axes are valid.

    Parameters
    ----------
    axes : str
        Axes to check.

    Returns
    -------
    bool
        Whether the axes are valid.
    """
    _axes = axes.upper()

    # length 0 and > 6
    if 0 > len(_axes) > 6:
        return False

    # all characters must be in REF_AXES = 'STZYXC'
    if not all(s in REF_AXES for s in _axes):
        return False

    # check for repeating characters
    for i, s in enumerate(_axes):
        if i != _axes.rfind(s):
            return False

    # prior: X and Y contiguous
    return ("XY" in _axes) or ("YX" in _axes)


def reshape_prediction(prediction: NDArray, axes: str, is_3d: bool) -> NDArray:
    """Reshape the prediction to match the input axes.

    The default axes of the model prediction is SC(Z)YX.

    Parameters
    ----------
    prediction : NDArray
        Prediction.
    axes : str
        Axes of the input data.
    is_3d : bool
        Whether the data is 3D.

    Returns
    -------
    NDArray
        Reshaped prediction.
    """
    # model outputs SC(Z)YX
    pred_axes = "SCZYX" if is_3d else "SCYX"

    # transpose the axes
    # TODO: during prediction T and S are merged. Check how to handle this
    input_axes = axes.replace("T", "S")
    remove_c, remove_s = False, False

    if "C" not in input_axes:
        # add C if missing
        input_axes = "C" + input_axes
        remove_c = True

    if "S" not in input_axes:
        # add S if missing
        input_axes = "S" + input_axes
        remove_s = True

    # TODO: check if all axes are present
    assert all(ax in input_axes for ax in pred_axes)

    indices = [pred_axes.index(ax) for ax in input_axes]
    prediction = np.transpose(prediction, indices)

    # remove S if not present in the input axes
    if remove_c:
        prediction = prediction[0]

    # remove C if not present in the input axes
    if remove_s:
        prediction = prediction[0]

    return prediction
