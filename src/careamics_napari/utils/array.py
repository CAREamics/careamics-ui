import numpy as np
from numpy.typing import NDArray


def get_prediction_samples(predictions: NDArray | list[NDArray]) -> list[NDArray]:
    """Make sure predictions are a list of numpy arrays and merge them if possible.

    Parameters
    ----------
        predictions (NDArray | list[NDArray]]): CAREamist predictions

    Returns
    -------
        list[NDArray]: a list of numpy arrays
    """
    if isinstance(predictions, list):
        # try to merge them into one array
        dims = np.array([a.shape[1:] for a in predictions])
        if (dims == dims[0]).all():
            return [np.stack(predictions, axis=0)]
        else:
            return predictions
    else:
        return [predictions]
