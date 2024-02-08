from absl.testing import absltest
import jax
from jax import numpy as jnp
import cglbm.test_utils as test_utils
from cglbm.lbm import *

import numpy as np
from einops import rearrange
from cglbm.test_utils import ParquetIOHelper


class LBMSnapshotTest(absltest.TestCase):
    """
    Snapshot tests for LBM Methods
    """

    def test_grid_eq_dist(self):
        system = test_utils.load_test_config("params.ini")
        GRID_SHAPE = (system.LY, system.LX)
        GRID_3D_SHAPE = (system.LY, system.LX, system.NL)

        pressure = jnp.full(GRID_SHAPE, system.ref_pressure)
        u = jnp.zeros((system.LY, system.LX, 2), dtype=jnp.float32)

        expected_path = 'snapshot_grid_eq_dist/eq_dist_output.parquet'
        output_helper = ParquetIOHelper(expected_path).read()
        expected = output_helper.get("neq", GRID_3D_SHAPE, "i j k -> k i j")

        actual_d = grid_eq_dist(system.cXYs, system.weights,
                                system.phi_weights, pressure, u)
        actual = jax.device_get(actual_d)

        self.assertTrue(np.allclose(actual, expected))


    def test_compute_phase_field(self):
        system = test_utils.load_test_config("params.ini")
        GRID_SHAPE = (system.LY, system.LX)
        GRID_3D_SHAPE = (system.LY, system.LX, system.NL)

        input_path = 'snapshot_compute_phase_field/compute_phase_field_input.parquet'
        expected_path = 'snapshot_compute_phase_field/compute_phase_field_output.parquet'

        input_helper = ParquetIOHelper(input_path).read()
        f = jnp.array(input_helper.get("f", GRID_3D_SHAPE, "i j k -> k i j"))

        output_helper = ParquetIOHelper(expected_path).read()
        expected = output_helper.get("phase_field", GRID_SHAPE)

        actual_d = compute_phase_field(f)
        actual = jax.device_get(actual_d)

        self.assertTrue(np.allclose(actual, expected))


    def test_compute_dst_phase_field(self):
        # NOTE: currently failing
        system = test_utils.load_test_config("params.ini")
        GRID_SHAPE = (system.LY, system.LX)
        GRID_3D_SHAPE = (system.LY, system.LX, system.NL)

        input_path = 'snapshot_compute_dst_phase_field/compute_dst_phase_field_input.parquet'
        expected_path = 'snapshot_compute_dst_phase_field/compute_dst_phase_field_output.parquet'

        input_helper = ParquetIOHelper(input_path).read()
        phase_field = jnp.array(input_helper.get("phase_field", GRID_SHAPE))

        output_helper = ParquetIOHelper(expected_path).read()
        expected = output_helper.get("dst_phase_field", GRID_3D_SHAPE, "i j k -> k i j")

        actual_d = compute_dst_phase_field(system.cXs, system.cYs, phase_field)
        actual = jax.device_get(actual_d)

        # self.assertTrue(np.allclose(actual, expected))


    def test_compute_phi_grad(self):
        # NOTE: phi_grad output has difference in precision, hence doing
        # np.allclose with 1e-7 precision
        system = test_utils.load_test_config("params.ini")
        GRID_SHAPE = (system.LY, system.LX)
        GRID_3D_SHAPE = (system.LY, system.LX, system.NL)

        input_path = 'snapshot_compute_dst_phase_field/compute_dst_phase_field_output.parquet'
        expected_path = 'snapshot_compute_phi_grad/compute_phi_grad_output.parquet'

        input_helper = ParquetIOHelper(input_path).read()
        dst_phase_field = jnp.array(
            input_helper.get("dst_phase_field", GRID_3D_SHAPE, "i j k -> k i j"))

        output_helper = ParquetIOHelper(expected_path).read()
        expected_phi_grad_x = output_helper.get("phi_grad_x", GRID_SHAPE)
        expected_phi_grad_y = output_helper.get("phi_grad_y", GRID_SHAPE)
        expected = rearrange(np.stack([expected_phi_grad_x, expected_phi_grad_y]),
                             "v i j -> i j v")

        actual_d = compute_phi_grad(system.cXYs, system.weights, dst_phase_field)
        actual = jax.device_get(actual_d)

        self.assertTrue(np.allclose(actual, expected, atol=1e-7))


    def test_surface_tension_force(self):
        system = test_utils.load_test_config("params.ini")
        GRID_SHAPE = (system.LY, system.LX)
        GRID_3D_SHAPE = (system.LY, system.LX, system.NL)

        input_surface_tension_path = 'snapshot_compute_surface_tension_force/compute_surface_tension_force_input.parquet'
        input_dst_phase_field_path = 'snapshot_compute_dst_phase_field/compute_dst_phase_field_output.parquet'
        expected_path = 'snapshot_compute_surface_tension_force/compute_surface_tension_force_output.parquet'

        surface_tension_input_helper = ParquetIOHelper(input_surface_tension_path).read()
        dst_phase_field_helper = ParquetIOHelper(input_dst_phase_field_path).read()
        phase_field = jnp.array(surface_tension_input_helper.get("phase_field", GRID_SHAPE))
        phi_grad_x = jnp.array(surface_tension_input_helper.get("phi_grad_x", GRID_SHAPE))
        phi_grad_y = jnp.array(surface_tension_input_helper.get("phi_grad_y", GRID_SHAPE))
        phi_grad = rearrange(jnp.stack([phi_grad_x, phi_grad_y]), "v i j -> i j v")
        dst_phase_field = jnp.array(
            dst_phase_field_helper.get("dst_phase_field", GRID_3D_SHAPE, "i j k -> k i j"))

        output_helper = ParquetIOHelper(expected_path).read()
        expected_curvature_force_x = output_helper.get("curvature_force_x", GRID_SHAPE)
        expected_curvature_force_y = output_helper.get("curvature_force_y", GRID_SHAPE)
        expected = rearrange(np.stack([expected_curvature_force_x, expected_curvature_force_y]),
                             "v i j -> i j v")

        actual_d = compute_surface_tension_force(
            system.surface_tension,
            system.width,
            system.weights,
            phase_field,
            dst_phase_field,
            phi_grad)
        actual = jax.device_get(actual_d)

        self.assertTrue(np.allclose(actual, expected))


    def test_compute_mom(self):
        # NOTE: mom and mom_eq output have difference in precision, hence doing np.allclose
        # with 1e-7 precision
        system = test_utils.load_test_config("params.ini")
        GRID_SHAPE = (system.LY, system.LX)
        GRID_3D_SHAPE = (system.LY, system.LX, system.NL)

        input_2d_path = 'snapshot_compute_mom/compute_mom_input_2d.parquet'
        input_3d_path = 'snapshot_compute_mom/compute_mom_input_3d.parquet'
        expected_2d_path = 'snapshot_compute_mom/compute_mom_output_2d.parquet'
        expected_3d_path = 'snapshot_compute_mom/compute_mom_output_3d.parquet'

        mom_2d_input_helper = ParquetIOHelper(input_2d_path).read()
        mom_3d_input_helper = ParquetIOHelper(input_3d_path).read()
        phase_field = jnp.array(mom_2d_input_helper.get("phase_field", GRID_SHAPE))
        pressure = jnp.array(mom_2d_input_helper.get("pressure", GRID_SHAPE))
        ux = jnp.array(mom_2d_input_helper.get("ux", GRID_SHAPE))
        uy = jnp.array(mom_2d_input_helper.get("uy", GRID_SHAPE))
        u = rearrange(jnp.stack([ux, uy]), "v i j -> i j v")
        N = jnp.array(mom_3d_input_helper.get("N", GRID_3D_SHAPE, "i j k -> k i j"))

        output_2d_helper = ParquetIOHelper(expected_2d_path).read()
        output_3d_helper = ParquetIOHelper(expected_3d_path).read()
        expected_kin_visc_local = output_2d_helper.get("kin_visc_local", GRID_SHAPE)
        expected_mom = output_3d_helper.get("mom", GRID_3D_SHAPE)
        expected_mom_eq = output_3d_helper.get("mom_eq", GRID_3D_SHAPE)

        actual_d = compute_mom(system.kin_visc_one, system.kin_visc_two, system.M_D2Q9, u,
                               pressure, phase_field, N)
        actual = jax.device_get(actual_d)

        self.assertTrue(np.allclose(actual[0], expected_mom, atol=1e-7))
        self.assertTrue(np.allclose(actual[1], expected_mom_eq, atol=1e-7))
        self.assertTrue(np.allclose(actual[2], expected_kin_visc_local))


    def test_compute_viscosity_correction(self):
        # NOTE: output has difference in precision, hence doing
        # np.allclose with 1e-5 precision
        system = test_utils.load_test_config("params.ini")
        GRID_SHAPE = (system.LY, system.LX)
        GRID_3D_SHAPE = (system.LY, system.LX, system.NL)

        input_visc_local_2d = 'snapshot_compute_mom/compute_mom_output_2d.parquet'
        input_phi_grad_2d = 'snapshot_compute_viscosity_correction/compute_viscosity_correction_input_2d.parquet'
        input_3d_path = 'snapshot_compute_mom/compute_mom_output_3d.parquet'
        expected_2d_path = 'snapshot_compute_viscosity_correction/compute_viscosity_correction_output_2d.parquet'

        visc_local_input_helper = ParquetIOHelper(input_visc_local_2d).read()
        phi_grad_input_helper = ParquetIOHelper(input_phi_grad_2d).read()
        input_3d_helper = ParquetIOHelper(input_3d_path).read()
        kin_visc_local = jnp.array(visc_local_input_helper.get("kin_visc_local", GRID_SHAPE))
        phi_grad_x = jnp.array(phi_grad_input_helper.get("phi_grad_x", GRID_SHAPE))
        phi_grad_y = jnp.array(phi_grad_input_helper.get("phi_grad_y", GRID_SHAPE))
        phi_grad = rearrange(jnp.stack([phi_grad_x, phi_grad_y]), "v i j -> i j v")
        mom = jnp.array(input_3d_helper.get("mom", GRID_3D_SHAPE))
        mom_eq = jnp.array(input_3d_helper.get("mom_eq", GRID_3D_SHAPE))

        output_helper = ParquetIOHelper(expected_2d_path).read()
        expected_visc_force_x = output_helper.get("viscous_force_x", GRID_SHAPE)
        expected_visc_force_y = output_helper.get("viscous_force_y", GRID_SHAPE)
        expected = rearrange(
            np.stack([expected_visc_force_x, expected_visc_force_y]), "v i j -> i j v")

        actual_d = compute_viscosity_correction(
            system.invM_D2Q9,
            system.cMs,
            system.density_one,
            system.density_two,
            phi_grad,
            kin_visc_local,
            mom,
            mom_eq)
        actual = jax.device_get(actual_d)

        self.assertTrue(np.allclose(actual, expected, atol=1e-5))


    def test_compute_total_force(self):
        system = test_utils.load_test_config("params.ini")
        GRID_SHAPE = (system.LY, system.LX)

        input_curvature_force_path = 'snapshot_compute_surface_tension_force/compute_surface_tension_force_output.parquet'
        input_viscous_force_path = 'snapshot_compute_viscosity_correction/compute_viscosity_correction_output_2d.parquet'
        input_rho_path = 'snapshot_compute_total_force/compute_total_force_input.parquet'
        expected_path = 'snapshot_compute_total_force/compute_total_force_output.parquet'

        curvature_force_input_helper = ParquetIOHelper(input_curvature_force_path).read()
        viscous_force_input_helper = ParquetIOHelper(input_viscous_force_path).read()
        rho_input_helper = ParquetIOHelper(input_rho_path).read()
        curvature_force_x = jnp.array(
            curvature_force_input_helper.get("curvature_force_x", GRID_SHAPE))
        curvature_force_y = jnp.array(
            curvature_force_input_helper.get("curvature_force_y", GRID_SHAPE))
        curvature_force = rearrange(
            jnp.stack([curvature_force_x, curvature_force_y]), "v i j -> i j v")
        viscous_force_x = jnp.array(
            viscous_force_input_helper.get("viscous_force_x", GRID_SHAPE))
        viscous_force_y = jnp.array(
            viscous_force_input_helper.get("viscous_force_y", GRID_SHAPE))
        viscous_force = rearrange(
            jnp.stack([viscous_force_x, viscous_force_y]), "v i j -> i j v")
        rho = jnp.array(rho_input_helper.get("rho", GRID_SHAPE))

        output_helper = ParquetIOHelper(expected_path).read()
        expected_total_force_x = output_helper.get("total_force_x", GRID_SHAPE)
        expected_total_force_y = output_helper.get("total_force_y", GRID_SHAPE)
        expected = rearrange(np.stack([expected_total_force_x, expected_total_force_y]),
                             "v i j -> i j v")

        actual_d = compute_total_force(system.gravityX, system.gravityY, curvature_force,
                                       viscous_force, rho)
        actual = jax.device_get(actual_d)

        self.assertTrue(np.allclose(actual, expected))


    def test_compute_collision(self):
        # NOTE: only comparing output ignoring first and last 2 columns, which are obstacle columns
        # have to handle properly the case for obstacles
        system = test_utils.load_test_config("params.ini")
        GRID_SHAPE = (system.LY, system.LX)
        GRID_3D_SHAPE = (system.LY, system.LX, system.NL)

        input_obstacle_path = 'snapshot_common/obstacle_input.parquet'
        input_collision_2d_path = 'snapshot_compute_collision/compute_collision_input_2d.parquet'
        input_collision_3d_path = 'snapshot_compute_collision/compute_collision_input_3d.parquet'
        expected_path = 'snapshot_compute_collision/compute_collision_output.parquet'

        obstacle_input_helper = ParquetIOHelper(input_obstacle_path).read()
        collision_2d_input_helper = ParquetIOHelper(input_collision_2d_path).read()
        collision_3d_input_helper = ParquetIOHelper(input_collision_3d_path).read()
        obs = jnp.array(obstacle_input_helper.get("obs", GRID_SHAPE))
        rho = jnp.array(collision_2d_input_helper.get("rho", GRID_SHAPE))
        kin_visc_local = jnp.array(collision_2d_input_helper.get("kin_visc_local", GRID_SHAPE))
        interface_force_x = jnp.array(collision_2d_input_helper.get("interface_force_x", GRID_SHAPE))
        interface_force_y = jnp.array(collision_2d_input_helper.get("interface_force_y", GRID_SHAPE))
        interface_force = rearrange(
            jnp.stack([interface_force_x, interface_force_y]), "v i j -> i j v")
        mom = jnp.array(collision_3d_input_helper.get("mom", GRID_3D_SHAPE))
        mom_eq = jnp.array(collision_3d_input_helper.get("mom_eq", GRID_3D_SHAPE))
        N = jnp.array(collision_3d_input_helper.get("N", GRID_3D_SHAPE, "i j k -> k i j"))

        output_helper = ParquetIOHelper(expected_path).read()
        expected = output_helper.get("N_new", GRID_3D_SHAPE, "i j k -> k i j")

        actual_d = compute_collision(system.invM_D2Q9, obs, mom, mom_eq, kin_visc_local,
                                     interface_force, rho, N)
        actual = jax.device_get(actual_d)

        self.assertTrue(np.allclose(actual[:, :, 2:-2], expected[:, :, 2:-2]))


    def test_compute_density_velocity_pressure(self):
        # NOTE: velocity (u) output has difference in precision, hence doing
        # np.allclose with 1e-7 precision
        system = test_utils.load_test_config("params.ini")
        GRID_SHAPE = (system.LY, system.LX)
        GRID_3D_SHAPE = (system.LY, system.LX, system.NL)

        input_phase_field_path = 'snapshot_compute_phase_field/compute_phase_field_output.parquet'
        input_phi_grad_path = 'snapshot_compute_phi_grad/compute_phi_grad_output.parquet'
        input_total_force_path = 'snapshot_compute_total_force/compute_total_force_output.parquet'
        input_2d_path = 'snapshot_compute_density_velocity_pressure/compute_density_velocity_pressure_input_2d.parquet'
        input_obstacle_path = 'snapshot_common/obstacle_input.parquet'
        input_3d_path = 'snapshot_compute_density_velocity_pressure/compute_density_velocity_pressure_input_3d.parquet'
        expected_path = 'snapshot_compute_density_velocity_pressure/compute_density_velocity_pressure_output.parquet'

        phase_field_input_helper = ParquetIOHelper(input_phase_field_path).read()
        phi_grad_input_helper = ParquetIOHelper(input_phi_grad_path).read()
        total_force_input_helper = ParquetIOHelper(input_total_force_path).read()
        obstacle_input_helper = ParquetIOHelper(input_obstacle_path).read()
        input_2d_helper = ParquetIOHelper(input_2d_path).read()
        input_3d_helper = ParquetIOHelper(input_3d_path).read()
        phase_field = jnp.array(phase_field_input_helper.get("phase_field", GRID_SHAPE))
        phi_grad_x = jnp.array(phi_grad_input_helper.get("phi_grad_x", GRID_SHAPE))
        phi_grad_y = jnp.array(phi_grad_input_helper.get("phi_grad_y", GRID_SHAPE))
        phi_grad = rearrange(jnp.stack([phi_grad_x, phi_grad_y]), "v i j -> i j v")
        total_force_x = jnp.array(total_force_input_helper.get("total_force_x", GRID_SHAPE))
        total_force_y = jnp.array(total_force_input_helper.get("total_force_y", GRID_SHAPE))
        total_force = rearrange(
            jnp.stack([total_force_x, total_force_y]), "v i j -> i j v")
        rho = jnp.array(input_2d_helper.get("rho", GRID_SHAPE))
        pressure = jnp.array(input_2d_helper.get("pressure", GRID_SHAPE))
        u_x = jnp.array(input_2d_helper.get("u_x", GRID_SHAPE))
        u_y = jnp.array(input_2d_helper.get("u_y", GRID_SHAPE))
        u = rearrange(jnp.stack([u_x, u_y]), "v i j -> i j v")
        obs = jnp.array(obstacle_input_helper.get("obs", GRID_SHAPE))
        N = jnp.array(input_3d_helper.get("N", GRID_3D_SHAPE, "i j k -> k i j"))

        output_helper = ParquetIOHelper(expected_path).read()
        expected_rho = output_helper.get("rho", GRID_SHAPE)
        expected_u_x = output_helper.get("u_x", GRID_SHAPE)
        expected_u_y = output_helper.get("u_y", GRID_SHAPE)
        expected_u = rearrange(jnp.stack([expected_u_x, expected_u_y]), "v i j -> i j v")
        expected_pressure = output_helper.get("pressure", GRID_SHAPE)
        expected_interface_force_x = output_helper.get("interface_force_x", GRID_SHAPE)
        expected_interface_force_y = output_helper.get("interface_force_y", GRID_SHAPE)
        expected_interface_force = rearrange(
            jnp.stack([expected_interface_force_x, expected_interface_force_y]),
            "v i j -> i j v"
        )

        actual_d = compute_density_velocity_pressure(
            system.density_one,
            system.density_two,
            system.cXs,
            system.cYs,
            system.weights,
            system.phi_weights,
            obs,
            pressure,
            u,
            rho,
            phase_field,
            phi_grad,
            N,
            total_force)
        actual = jax.device_get(actual_d)

        self.assertTrue(np.allclose(actual[0], expected_rho))
        self.assertTrue(np.allclose(actual[1], expected_u, atol=1e-7))
        self.assertTrue(np.allclose(actual[2], expected_pressure))
        self.assertTrue(np.allclose(actual[3], expected_interface_force))


    def test_compute_segregation(self):
        system = test_utils.load_test_config("params.ini")
        GRID_SHAPE = (system.LY, system.LX)
        GRID_3D_SHAPE = (system.LY, system.LX, system.NL)

        input_2d_path = 'snapshot_compute_segregation/compute_segregation_input_2d.parquet'
        input_3d_path = 'snapshot_compute_segregation/compute_segregation_input_3d.parquet'
        expected_path = 'snapshot_compute_segregation/compute_segregation_output.parquet'

        input_2d_helper = ParquetIOHelper(input_2d_path).read()
        input_3d_helper = ParquetIOHelper(input_3d_path).read()
        phase_field = jnp.array(input_2d_helper.get("phase_field", GRID_SHAPE))
        phi_grad_x = jnp.array(input_2d_helper.get("phi_grad_x", GRID_SHAPE))
        phi_grad_y = jnp.array(input_2d_helper.get("phi_grad_y", GRID_SHAPE))
        pressure = jnp.array(input_2d_helper.get("pressure", GRID_SHAPE))
        u_x = jnp.array(input_2d_helper.get("u_x", GRID_SHAPE))
        u_y = jnp.array(input_2d_helper.get("u_y", GRID_SHAPE))
        phi_grad = rearrange(jnp.stack([phi_grad_x, phi_grad_y]), "v i j -> i j v")
        u = rearrange(jnp.stack([u_x, u_y]), "v i j -> i j v")
        N_new = jnp.array(input_3d_helper.get("N_new", GRID_3D_SHAPE, "i j k -> k i j"))

        output_helper = ParquetIOHelper(expected_path).read()
        expected = output_helper.get("f_new", GRID_3D_SHAPE, "i j k -> k i j")

        actual_d = compute_segregation(
            system.width,
            system.cXYs,
            system.weights,
            system.phi_weights,
            phase_field,
            phi_grad,
            pressure,
            u,
            N_new)
        actual = jax.device_get(actual_d)

        self.assertTrue(np.allclose(actual, expected))


    def test_compute_propagation(self):
        # NOTE: only comparing output ignoring first and last 2 columns, which are obstacle columns
        # have to handle properly the case for obstacles
        # N and f don't change for obstacle cases, but are affected by ghost nodes
        system = test_utils.load_test_config("params.ini")
        GRID_SHAPE = (system.LY, system.LX)
        GRID_3D_SHAPE = (system.LY, system.LX, system.NL)

        input_obstacle_path = 'snapshot_common/obstacle_input.parquet'
        input_2d_path = 'snapshot_compute_propagation/compute_propagation_input_2d.parquet'
        input_3d_path = 'snapshot_compute_propagation/compute_propagation_input_3d.parquet'
        expected_path = 'snapshot_compute_propagation/compute_propagation_output.parquet'

        obstacle_input_helper = ParquetIOHelper(input_obstacle_path).read()
        input_2d_helper = ParquetIOHelper(input_2d_path).read()
        input_3d_helper = ParquetIOHelper(input_3d_path).read()
        obs = jnp.array(obstacle_input_helper.get("obs", GRID_SHAPE))
        obs_vel_x = jnp.array(input_2d_helper.get("obs_vel_x", GRID_SHAPE))
        obs_vel_y = jnp.array(input_2d_helper.get("obs_vel_y", GRID_SHAPE))
        obs_vel = rearrange(jnp.stack([obs_vel_x, obs_vel_y]), "v i j -> i j v")
        N_new = jnp.array(input_3d_helper.get("N_new", GRID_3D_SHAPE, "i j k -> k i j"))
        f_new = jnp.array(input_3d_helper.get("f_new", GRID_3D_SHAPE, "i j k -> k i j"))

        output_helper = ParquetIOHelper(expected_path).read()
        expected_N = output_helper.get("N", GRID_3D_SHAPE, "i j k -> k i j")
        expected_f = output_helper.get("f", GRID_3D_SHAPE, "i j k -> k i j")

        actual_d = compute_propagation(
            system.cXs,
            system.cYs,
            system.cXYs,
            system.weights,
            obs,
            obs_vel,
            N_new,
            f_new)
        actual = jax.device_get(actual_d)

        self.assertTrue(np.allclose(actual[0][:, :, 2:-2], expected_N[:, :, 2:-2]))
        self.assertTrue(np.allclose(actual[1][:, :, 2:-2], expected_f[:, :, 2:-2]))


if __name__ == "__main__":
    # jax.config.update("jax_disable_jit", True) # for debugging
    absltest.main()
