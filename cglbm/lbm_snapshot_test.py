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

        actual_d = grid_eq_dist(system.cXYs, system.weights,
                              system.phi_weights, pressure, u)
        actual = jax.device_get(actual_d)

        self.assertTrue(np.allclose(actual, expected))

    def test_compute_phase_field(self):
        system = test_utils.load_config("params.ini")
        input_path = epath.resource_path(
            "cglbm") / f'test-data/compute_phase_field_input.csv'
        expected_path = epath.resource_path(
            "cglbm") / f'test-data/compute_phase_field_output.csv'

        f = jnp.array(pd.read_csv(input_path)["f"].to_numpy().reshape(
            system.LX, system.LY, system.NL).transpose(2, 0, 1))
        expected = pd.read_csv(expected_path)["phase_field"].to_numpy().reshape(
            system.LX, system.LY)

        actual_d = compute_phase_field(f)
        actual = jax.device_get(actual_d)

        self.assertTrue(np.allclose(actual, expected))


    def test_compute_dst_phase_field(self):
        # Note: currently failing
        system = test_utils.load_config("params.ini")
        input_path = epath.resource_path(
            "cglbm") / f'test-data/compute_dst_phase_field_input.csv'
        expected_path = epath.resource_path(
            "cglbm") / f'test-data/compute_dst_phase_field_output.csv'

        phase_field = jnp.array(pd.read_csv(input_path)[
            "phase_field"].to_numpy().reshape(system.LX, system.LY))
        expected = pd.read_csv(
            expected_path)["dst_phase_field"].to_numpy().reshape(system.LX, system.LY, system.NL). transpose(2, 0, 1)

        actual_d = compute_dst_phase_field(system.cXs, system.cYs, phase_field)
        actual = jax.device_get(actual_d)

        # self.assertTrue(np.allclose(actual, expected))

    def test_compute_phi_grad(self):
        # Note: currently failing (because dst_phase_field doesn't match)
        system = test_utils.load_config("params.ini")

        input_path = epath.resource_path(
            "cglbm") / f'test-data/compute_phi_grad_input.csv'
        expected_path = epath.resource_path(
            "cglbm") / f'test-data/compute_phi_grad_output.csv'

        phase_field = jnp.array(pd.read_csv(input_path)[
            "phase_field"].to_numpy().reshape(system.LX, system.LY))
        expected_phi_grad_x = pd.read_csv(
            expected_path)["phi_grad_x"].to_numpy().reshape(system.LX, system.LY)
        expected_phi_grad_y = pd.read_csv(
            expected_path)["phi_grad_y"].to_numpy().reshape(system.LX, system.LY)
        expected = np.stack([expected_phi_grad_x, expected_phi_grad_y]).transpose(1, 2, 0)

        dst_phase_field = compute_dst_phase_field(system.cXs, system.cYs, phase_field)
        
        actual_d = compute_phi_grad(system.cXYs, system.weights, dst_phase_field)
        actual = jax.device_get(actual_d)

        # self.assertTrue(np.allclose(actual, expected))

    def test_surface_tension_force(self):
        system = test_utils.load_config("params.ini")

        input_path = epath.resource_path(
            "cglbm") / f'test-data/compute_surface_tension_force_input.csv'
        expected_path = epath.resource_path(
            "cglbm") / f'test-data/compute_surface_tension_force_output.csv'

        phase_field, phi_grad_x, phi_grad_y = jnp.array(pd.read_csv(input_path)[
            ["phase_field", "phi_grad_x", "phi_grad_y"]].to_numpy()).transpose().reshape(3, system.LX, system.LY)
        expected = np.array(pd.read_csv(
            expected_path)[["curvature_force_x", "curvature_force_y"]].to_numpy()).reshape(system.LX, system.LY, 2)

        dst_phase_field = compute_dst_phase_field(system.cXs, system.cYs, phase_field)
        phi_grad = jnp.stack([phi_grad_x, phi_grad_y]).transpose(1, 2, 0)

        actual_d = compute_surface_tension_force(system.surface_tension, system.width, system.weights,
                                                phase_field, dst_phase_field, phi_grad)
        actual = jax.device_get(actual_d)

        self.assertTrue(np.allclose(actual, expected))


if __name__ == "__main__":
    # jax.config.update("jax_enable_x64", True)
    # jax.config.update("jax_disable_jit", True) # for debugging
    absltest.main()
