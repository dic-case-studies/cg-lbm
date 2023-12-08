from absl.testing import absltest
import jax
from jax import numpy as jnp
import cglbm.test_utils as test_utils
from cglbm.lbm import *

from etils import epath
import pandas as pd


class LBMSnapshotTest(absltest.TestCase):
    """
    Snapshot tests for LBM Methods
    """

    def test_grid_eq_dist(self):
        system = test_utils.load_config("params.ini")

        pressure = jnp.full((system.LX, system.LY), system.ref_pressure)
        u = jnp.zeros((system.LX, system.LY, 2), dtype=jnp.float32)
        actual = grid_eq_dist(system.cXYs, system.weights, system.phi_weights, pressure, u)

        neq_path = epath.resource_path("cglbm") / f'test-data/neq.csv'
        expected_neq = pd.read_csv(neq_path).to_numpy()
        expected = expected_neq.reshape(system.LX, system.LY, system.NL).transpose(2, 0, 1)

        print(len(np.argwhere((expected - actual) != 0)))
        self.assertTrue(np.allclose(actual, expected))


    def test_compute_phi_grad(self):
        # Note: currently failing
        jax.config.update("jax_enable_x64", True)

        system = test_utils.load_config("params.ini")

        input_path = epath.resource_path("cglbm") / f'test-data/phaseField.csv'
        phase_field = pd.read_csv(input_path).to_numpy().reshape(100, 100)
        dst_phase_field = compute_dst_phase_field(system.cXs, system.cYs, phase_field)

        actual = compute_phi_grad(system.cXYs, system.weights, dst_phase_field)

        phi_grad_x_path = epath.resource_path("cglbm") / f'test-data/phiGrad_x.csv'
        phi_grad_y_path = epath.resource_path("cglbm") / f'test-data/phiGrad_y.csv'
        expected_phi_grad_x = pd.read_csv(phi_grad_x_path).to_numpy().reshape(100, 100)
        expected_phi_grad_y = pd.read_csv(phi_grad_y_path).to_numpy().reshape(100, 100)
        expected = np.stack([expected_phi_grad_x, expected_phi_grad_y]).transpose(1, 2, 0)
        
        print(len(np.argwhere((expected - actual) > 0)))
        self.assertTrue(np.allclose(actual, expected))


if __name__ == "__main__":
    absltest.main()
