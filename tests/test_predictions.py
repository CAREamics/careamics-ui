# from src.careamics_napari.training_plugin import TrainingPlugin
import contextlib
import logging
import sys
from itertools import combinations

import numpy as np
from careamics import CAREamist

# disable logging
from careamics.careamist import logger
from careamics.config import create_n2v_configuration

from careamics_napari.utils.axes_utils import reshape_prediction

logger.setLevel("ERROR")
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.FATAL)


# nostdout from https://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto
class DummyFile:
    def write(self, x):
        pass


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


def generate_combinations_and_rotations(s):
    # generate all combinations
    combinations_list = []
    for r in range(1, len(s) + 1):
        combinations_list.extend(["".join(comb) for comb in combinations(s, r)])

    # generate all rotations
    rotations = set()
    for i in range(len(s)):
        rotated = s[i:] + s[:i]
        rotations.add(rotated)

    # combine results
    all_results = set(combinations_list)
    for rot in rotations:
        for r in range(1, len(rot) + 1):
            all_results.update(["".join(comb) for comb in combinations(rot, r)])

    # add an empty
    all_results.add("")

    return sorted(all_results)


augmentation = generate_combinations_and_rotations("TZC")
for ax in augmentation:
    test_axes = ax + "YX"
    n_channels = 1
    shape = []
    for ax in test_axes:
        if ax == "S":
            shape.append(2)
        elif ax == "T":
            shape.append(4)
        elif ax == "C":
            shape.append(3)
            n_channels = 3
        else:
            shape.append(16)

    pred_data = np.random.randint(0, 255, shape).astype(np.float32)
    with nostdout():
        # create a configuration
        config = create_n2v_configuration(
            experiment_name=f"N2V_{test_axes}",
            data_type="array",
            axes=test_axes,
            n_channels=n_channels,
            patch_size=[8, 8, 8] if "Z" in test_axes else [8, 8],
            batch_size=1,
            num_epochs=1,
            masked_pixel_percentage=2,
        )

        # instantiate a careamist
        careamist = CAREamist(config)
        careamist.cfg.data_config.set_means_and_stds(
            [127.0] * n_channels, [75.0] * n_channels
        )

        predction = careamist.predict(source=pred_data)
    if isinstance(predction, list):
        predction = np.concatenate(predction, axis=0)

    pred = reshape_prediction(predction, test_axes, "Z" in test_axes)

    assert pred_data.shape == pred.shape, (
        f"Prediction shape {pred_data.shape} != {predction.shape} for axes {test_axes}"
    )
