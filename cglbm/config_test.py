"""Tests for config"""
from cglbm.environment import System
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
import cglbm.test_utils as utils


class ConfigTest(parameterized.TestCase):
    def test_read_config(self):
        sys = utils.load_config("params.ini")
        np.testing.assert_equal(sys.LX, 100)
        np.testing.assert_almost_equal(sys.gravityX, 1e-6)


if __name__ == "__main__":
    absltest.main()
