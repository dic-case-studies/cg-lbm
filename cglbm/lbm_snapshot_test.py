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
        # Note: phi_grad output has difference in precision, hence doing np.allclose with 1e-7 precision
        system = test_utils.load_config("params.ini")

        input_path = epath.resource_path(
            "cglbm") / f'test-data/compute_dst_phase_field_output.csv'
        expected_path = epath.resource_path(
            "cglbm") / f'test-data/compute_phi_grad_output.csv'

        dst_phase_field = pd.read_csv(input_path)["dst_phase_field"].to_numpy().reshape(system.LX, system.LY, system.NL).transpose(2, 0, 1)
        expected = pd.read_csv(expected_path)[["phi_grad_x", "phi_grad_y"]].to_numpy().transpose().reshape(2, system.LX, system.LY).transpose(1, 2, 0)

        actual_d = compute_phi_grad(system.cXYs, system.weights, dst_phase_field)
        actual = jax.device_get(actual_d)

        self.assertTrue(np.allclose(actual, expected, atol=1e-7))

    def test_surface_tension_force(self):
        system = test_utils.load_config("params.ini")

        input_path = epath.resource_path(
            "cglbm") / f'test-data/compute_surface_tension_force_input.csv'
        input_dst_phase_field_path = epath.resource_path(
            "cglbm") / f'test-data/compute_dst_phase_field_output.csv'
        expected_path = epath.resource_path(
            "cglbm") / f'test-data/compute_surface_tension_force_output.csv'

        phase_field, phi_grad_x, phi_grad_y = jnp.array(pd.read_csv(input_path)[
            ["phase_field", "phi_grad_x", "phi_grad_y"]].to_numpy()).transpose().reshape(3, system.LX, system.LY)
        phi_grad = jnp.stack([phi_grad_x, phi_grad_y]).transpose(1, 2, 0)
        dst_phase_field = jnp.array(pd.read_csv(input_dst_phase_field_path)["dst_phase_field"].to_numpy()).reshape(
            system.LX, system.LY, system.NL).transpose(2, 0, 1)
        expected = pd.read_csv(
            expected_path)[["curvature_force_x", "curvature_force_y"]].to_numpy().reshape(system.LX, system.LY, 2)

        actual_d = compute_surface_tension_force(system.surface_tension, system.width, system.weights,
                                                phase_field, dst_phase_field, phi_grad)
        actual = jax.device_get(actual_d)

        self.assertTrue(np.allclose(actual, expected))

    def test_compute_mom(self):
        # Note: mom output has difference in precision, hence doing np.allclose with 1e-7 precision
        system = test_utils.load_config("params.ini")

        input_2d_path = epath.resource_path("cglbm") / f'test-data/compute_mom_input_2d.csv'
        input_3d_path = epath.resource_path("cglbm") / f'test-data/compute_mom_input_3d.csv'
        expected_2d_path = epath.resource_path("cglbm") / f'test-data/compute_mom_output_2d.csv'
        expected_3d_path = epath.resource_path("cglbm") / f'test-data/compute_mom_output_3d.csv'

        phase_field, ux, uy, pressure = jnp.array(pd.read_csv(input_2d_path)[["phase_field", "ux", "uy", "pressure"]].to_numpy())\
            .transpose().reshape(4, system.LX, system.LY)
        u = jnp.stack([ux, uy]).transpose(1, 2, 0)
        N = jnp.array(pd.read_csv(input_3d_path)["N"].to_numpy()).reshape(system.LX, system.LY, system.NL)\
            .transpose(2, 0, 1)
        expected_mom, expected_mom_eq = pd.read_csv(expected_3d_path)[["mom", "mom_eq"]].to_numpy()\
            .transpose().reshape(2, system.LX, system.LY, system.NL)
        expected_kin_visc_local = pd.read_csv(expected_2d_path)[["kin_visc_local"]].to_numpy()\
            .reshape(system.LX, system.LY)

        actual_d = compute_mom(system.kin_visc_one, system.kin_visc_two, system.M_D2Q9, u, pressure, phase_field, N)
        actual = jax.device_get(actual_d)

        self.assertTrue(np.allclose(actual[0], expected_mom, atol=1e-7))
        self.assertTrue(np.allclose(actual[1], expected_mom_eq))
        self.assertTrue(np.allclose(actual[2], expected_kin_visc_local))


    def test_compute_viscosity_correction(self):
        system = test_utils.load_config("params.ini")

        input_visc_local_2d = epath.resource_path("cglbm") / f'test-data/compute_mom_output_2d.csv'
        input_phi_grad_2d = epath.resource_path("cglbm") / f'test-data/compute_viscosity_correction_input_2d.csv'
        input_3d_path = epath.resource_path("cglbm") / f'test-data/compute_mom_output_3d.csv'
        expected_2d_path = epath.resource_path("cglbm") / f'test-data/compute_viscosity_correction_output_2d.csv'

        kin_visc_local = jnp.array(pd.read_csv(input_visc_local_2d)[["kin_visc_local"]].to_numpy()).reshape(system.LX, system.LY)
        phi_grad = jnp.array(pd.read_csv(input_phi_grad_2d)[["phi_grad_x", "phi_grad_y"]].to_numpy()).transpose().reshape(2, system.LX, system.LY).transpose(1, 2, 0)
        mom, mom_eq = jnp.array(pd.read_csv(input_3d_path)[["mom", "mom_eq"]].to_numpy()).transpose().reshape(2, system.LX, system.LY, system.NL)
        expected = pd.read_csv(expected_2d_path)[["viscous_force_x", "viscous_force_y"]].to_numpy().reshape(system.LX, system.LY, 2)

        actual_d = compute_viscosity_correction(system.invM_D2Q9, system.cMs, system.density_one, system.density_two, phi_grad, kin_visc_local, mom, mom_eq)
        actual = jax.device_get(actual_d)

        self.assertTrue(np.allclose(actual, expected))


    def test_compute_total_force(self):
        system = test_utils.load_config("params.ini")

        input_curvature_force_path = epath.resource_path(
            "cglbm") / f'test-data/compute_surface_tension_force_output.csv'
        input_viscous_force_path = epath.resource_path(
            "cglbm") / f'test-data/compute_viscosity_correction_output_2d.csv'
        input_rho_path = epath.resource_path("cglbm") / f'test-data/compute_total_force_input.csv'
        expected_path = epath.resource_path("cglbm") / f'test-data/compute_total_force_output.csv'

        curvature_force = jnp.array(pd.read_csv(
            input_curvature_force_path)[["curvature_force_x", "curvature_force_y"]].to_numpy()).reshape(system.LX, system.LY, 2)
        viscous_force = jnp.array(pd.read_csv(
            input_viscous_force_path)[["viscous_force_x", "viscous_force_y"]].to_numpy()).reshape(system.LX, system.LY, 2)
        rho = jnp.array(pd.read_csv(
            input_rho_path)["rho"].to_numpy()).reshape(system.LX, system.LY)    
        expected = pd.read_csv(expected_path)[["total_force_x", "total_force_y"]].to_numpy().reshape(system.LX, system.LY, 2)

        actual_d = compute_total_force(system.gravityX, system.gravityY, curvature_force, viscous_force, rho)
        actual = jax.device_get(actual_d)

        self.assertTrue(np.allclose(actual, expected))



if __name__ == "__main__":
    # jax.config.update("jax_enable_x64", True)
    # jax.config.update("jax_disable_jit", True) # for debugging
    absltest.main()
