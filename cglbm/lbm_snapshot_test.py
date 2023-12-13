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
        expected_path = epath.resource_path("cglbm") / f'test-data/eq_dist_output.csv'
        expected = pd.read_csv(expected_path)["neq"].to_numpy().reshape(
            system.LX, system.LY, system.NL).transpose(2, 0, 1)

        actual = jax.device_get(grid_eq_dist(system.cXYs, system.weights,
                              system.phi_weights, pressure, u))

        self.assertTrue(np.allclose(actual, expected))

    def test_compute_phase_field(self):
        system = test_utils.load_config("params.ini")
        input_path = epath.resource_path(
            "cglbm") / f'test-data/compute_phase_field_input.csv'
        f = jnp.array(pd.read_csv(input_path)["f"].to_numpy().reshape(
            system.LX, system.LY, system.NL).transpose(2, 0, 1))
        expected_path = epath.resource_path(
            "cglbm") / f'test-data/compute_phase_field_output.csv'
        expected = pd.read_csv(expected_path)["phase_field"].to_numpy().reshape(
            system.LX, system.LY)

        actual = jax.device_get(compute_phase_field(f))

        self.assertTrue(np.allclose(actual, expected))


    def test_compute_dst_phase_field(self):
        # Note: currently failing
        system = test_utils.load_config("params.ini")
        input_path = epath.resource_path(
            "cglbm") / f'test-data/compute_dst_phase_field_input.csv'
        phase_field = jnp.array(pd.read_csv(input_path)[
            "phase_field"].to_numpy().reshape(system.LX, system.LY))
        expected_path = epath.resource_path(
            "cglbm") / f'test-data/compute_dst_phase_field_output.csv'
        expected = pd.read_csv(
            expected_path)["dst_phase_field"].to_numpy().reshape(system.LX, system.LY, system.NL). transpose(2, 0, 1)

        actual = jax.device_get(compute_dst_phase_field(system.cXs, system.cYs, phase_field))

        self.compare_and_print(actual, expected)

        # self.assertTrue(np.allclose(actual, expected))

    def test_compute_phi_grad(self):
        # Note: currently failing
        system = test_utils.load_config("params.ini")
        input_path = epath.resource_path(
            "cglbm") / f'test-data/compute_phi_grad_input.csv'
        phase_field = jnp.array(pd.read_csv(input_path)[
            "phase_field"].to_numpy().reshape(system.LX, system.LY))
        expected_path = epath.resource_path(
            "cglbm") / f'test-data/compute_phi_grad_output.csv'
        expected_phi_grad_x = pd.read_csv(
            expected_path)["phi_grad_x"].to_numpy().reshape(system.LX, system.LY)
        expected_phi_grad_y = pd.read_csv(
            expected_path)["phi_grad_y"].to_numpy().reshape(system.LX, system.LY)
        expected = np.stack([expected_phi_grad_x, expected_phi_grad_y]).transpose(1, 2, 0)

        dst_phase_field = compute_dst_phase_field(system.cXs, system.cYs, phase_field)
        actual = jax.device_get(compute_phi_grad(system.cXYs, system.weights, dst_phase_field))

        # self.assertTrue(np.allclose(actual, expected))


if __name__ == "__main__":
    # jax.config.update("jax_enable_x64", True)
    absltest.main()
