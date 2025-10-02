import pytest
from careamics import CAREamist
from careamics.config import Configuration

from careamics_napari.careamics_utils import (
    get_default_care_config,
    get_default_n2n_config,
    get_default_n2v_config,
)


@pytest.mark.parametrize("algorithm", ["n2v", "care", "n2n"])
def test_creating_configuration(algorithm):
    """Test creating a configuration and initialize careamist with it."""
    config_methods = {
        "n2v": get_default_n2v_config,
        "care": get_default_care_config,
        "n2n": get_default_n2n_config,
    }
    config_fn = config_methods.get(algorithm)
    assert config_fn is not None

    config = config_fn()
    assert isinstance(config, Configuration)

    careamist = CAREamist(source=config)
    assert isinstance(careamist, CAREamist)
