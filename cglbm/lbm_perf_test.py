from absl.testing import absltest
import jax
from jax import numpy as jnp
import cglbm.test_utils as test_utils
from cglbm.lbm import *


class LBMPerfTest(absltest.TestCase):
    """Performance tests for LBM Methods
    Note:These are for reference actual numbers will go down after fusion
    """

    def test_perf_eq_dist_phase_field(self):
        system = test_utils.load_test_config("params.ini")

        def init_fn(rng):
            LX = system.LX
            LY = system.LY
            rng1, rng2 = jax.random.split(rng, 2)
            phi = jax.random.normal(rng1, (LY, LX))
            u = jax.random.normal(rng2, (LY, LX, 2))
            return {
                "phi": phi,
                "u": u
            }

        def step_fn(state):
            return eq_dist_phase_field(system.cXYs, system.weights, state["phi"], state["u"])

        test_utils.benchmark("benchmark eq dist phase field", init_fn, step_fn)


    def test_perf_wetting_boundary_condition_solid(self):
        system = test_utils.load_test_config("params.ini")
        width = 4.0
        contact_angle = 45

        def init_fn(rng):
            LX = system.LX
            LY = system.LY
            rng = jax.random.split(rng, 3)
            phase_field = jax.random.uniform(rng[0], (LY, LX))
            obs_indices = tuple(jax.random.randint(rng[1], (60, 2), 0, min(LX, LY)).T)
            surface_normals = jax.random.uniform(rng[2], (60, 2))

            return {
                "phase_field": phase_field,
                "surface_normals": surface_normals,
                "obs_indices": obs_indices
            }

        def step_fn(state):
            return wetting_boundary_condition_solid(
                width,
                contact_angle,
                state["obs_indices"],
                state["surface_normals"],
                state["phase_field"]
            )

        test_utils.benchmark("benchmark wetting boundary condition", init_fn, step_fn)


    def test_perf_grid_eq_dist(self):
        system = test_utils.load_test_config("params.ini")

        def init_fn(rng):
            LX = system.LX
            LY = system.LY
            rng1, rng2 = jax.random.split(rng, 2)
            pressure = jax.random.normal(rng1, (LY, LX))
            u = jax.random.normal(rng2, (LY, LX, 2))
            return {
                "pressure": pressure,
                "u": u
            }

        def step_fn(state):
            return grid_eq_dist(system.cXYs, system.weights, system.phi_weights, state["pressure"], state["u"])

        test_utils.benchmark("benchmark grid eq dist", init_fn, step_fn)

    
    def test_perf_eq_dist(self):
        system = test_utils.load_test_config("params.ini")

        def init_fn(rng):
            rng1, rng2 = jax.random.split(rng, 2)
            phi = jax.random.normal(rng1, (1,))
            u = jax.random.normal(rng2, (2,))
            return {
                "phi": phi,
                "u": u
            }

        def step_fn(state):
            return eq_dist(system.cXYs, system.weights, system.phi_weights, state["phi"], state["u"])

        test_utils.benchmark("benchmark eq dist", init_fn, step_fn)


    def test_perf_compute_phase_field(self):
        system = test_utils.load_test_config("params.ini")

        def init_fn(rng):
            LX = system.LX
            LY = system.LY
            rngs = jax.random.split(rng, 2)
            phase_field = jax.random.uniform(rngs[0],(LY, LX))
            f = jax.random.normal(rngs[1], (9, LY, LX))
            obs_mask = jnp.zeros((LY, LX), dtype=bool)
            return {
                "phase_field": phase_field,
                "f": f,
                "obs_mask": obs_mask
            }

        def step_fn(state):
            return compute_phase_field(state["phase_field"], state["f"], state["obs_mask"])

        test_utils.benchmark("benchmark compute phase field", init_fn, step_fn)


    def test_perf_compute_dst_phase_field(self):
        system = test_utils.load_test_config("params.ini")

        def init_fn(rng):
            LX = system.LX
            LY = system.LY
            rng1, _ = jax.random.split(rng, 2)
            phi = jax.random.normal(rng1, (LY, LX))
            return {
                "phi": phi
            }

        def step_fn(state):
            return compute_dst_phase_field(system.cXs, system.cYs, state["phi"])

        test_utils.benchmark("benchmark compute dst phase field", init_fn, step_fn)


    def test_perf_compute_phi_grad(self):
        system = test_utils.load_test_config("params.ini")

        def init_fn(rng):
            LX = system.LX
            LY = system.LY
            rng1, _ = jax.random.split(rng, 2)
            dst_phase_field = jax.random.normal(rng1, (9, LY, LX))
            return {
                "dst_phase_field": dst_phase_field
            }

        def step_fn(state):
            return compute_phi_grad(system.cXYs, system.weights, state["dst_phase_field"])

        test_utils.benchmark("benchmark compute phi grad", init_fn, step_fn)


    def test_perf_surface_tension_force(self):
        system = test_utils.load_test_config("params.ini")

        def init_fn(rng):
            LX = system.LX
            LY = system.LY
        
            rngs = jax.random.split(rng, 3)

            phase_field = jax.random.normal(rngs[0], (LY, LX))
            dst_phase_field = jax.random.normal(rngs[1], (9, LY, LX))
            phi_grad = jax.random.normal(rngs[2], (LY, LX, 2))

            return {
                "surface_tension": system.surface_tension,
                "width": system.width,
                "weights": system.weights,
                "phase_field": phase_field,
                "dst_phase_field": dst_phase_field,
                "phi_grad": phi_grad,
            }

        def step_fn(state):
            return compute_surface_tension_force(
                state["surface_tension"],
                state["width"],
                state["weights"],
                state["phase_field"],
                state["dst_phase_field"],
                state["phi_grad"]
            )

        test_utils.benchmark("benchmark surface tension force", init_fn, step_fn)


    def test_perf_compute_mom(self):
        system = test_utils.load_test_config("params.ini")

        def init_fn(rng):
            LX = system.LX
            LY = system.LY
            rngs = jax.random.split(rng, 4)

            phase_field = jax.random.normal(rngs[0], (LY, LX))
            u = jax.random.normal(rngs[1], (LY, LX, 2))
            pressure = jax.random.normal(rngs[2], (LY, LX))
            N = jax.random.normal(rngs[3], (9, LY, LX))

            return {
                "kin_visc_one": system.kin_visc_one,
                "kin_visc_two": system.kin_visc_two,
                "M_D2Q9": system.M_D2Q9,
                "u": u,
                "pressure": pressure,
                "phase_field": phase_field,
                "N": N
            }
        
        def step_fn(state):
            return compute_mom(
                state["kin_visc_one"],
                state["kin_visc_two"],
                state["M_D2Q9"],
                state["u"],
                state["pressure"],
                state["phase_field"],
                state["N"]
            )

        test_utils.benchmark("benchmark compute mom", init_fn, step_fn)


    def test_perf_compute_viscosity_correction(self):
        system = test_utils.load_test_config("params.ini")

        def init_fn(rng):
            LX = system.LX
            LY = system.LY
            rngs = jax.random.split(rng, 5)

            return {
                "invM_D2Q9": system.invM_D2Q9,
                "cMs": system.cMs,
                "density_one": system.density_one,
                "density_two": system.density_two,

                "phi_grad": jax.random.normal(rngs[0], (LY, LX, 2)),
                "kin_visc_local": jax.random.normal(rngs[1], (LY, LX)),
                "mom": jax.random.normal(rngs[2], (LY, LX, 9)),
                "mom_eq": jax.random.normal(rngs[3], (LY, LX, 9)),
            }

        def step_fn(state):
            return compute_viscosity_correction(
                state["invM_D2Q9"],
                state["cMs"],
                state["density_one"],
                state["density_two"],
                state["phi_grad"],
                state["kin_visc_local"],
                state["mom"],
                state["mom_eq"]
            )

        test_utils.benchmark("benchmark compute viscosity correction", init_fn, step_fn)


    def test_perf_compute_collision(self):
        system = test_utils.load_test_config("params.ini")

        def init_fn(rng):
            LX = system.LX
            LY = system.LY
            # TODO: Modify the way this is initialized, or set a well defined obstacle (wall, cylinder, etc)
            obs = jnp.zeros((LY, LX), dtype=bool)
            rngs = jax.random.split(rng, 7)

            return {
                "invM_D2Q9": system.invM_D2Q9,
                "obs": obs,

                "mom": jax.random.normal(rngs[1], (LY, LX, 9)),
                "mom_eq": jax.random.normal(rngs[2], (LY, LX, 9)),
                "kin_visc_local": jax.random.normal(rngs[3], (LY, LX)),
                "interface_force": jax.random.normal(rngs[4], (LY, LX, 2)),
                "rho": jax.random.normal(rngs[5], (LY, LX)),
                "N": jax.random.normal(rngs[6], (9, LY, LX))
            }

        def step_fn(state):
            return compute_collision(
                state["invM_D2Q9"],
                state["obs"],
                state["mom"],
                state["mom_eq"],
                state["kin_visc_local"],
                state["interface_force"],
                state["rho"],
                state["N"]
            )

        test_utils.benchmark("benchmark compute collision", init_fn, step_fn)


    def test_perf_compute_propagation(self):
        system = test_utils.load_test_config("params.ini")

        def init_fn(rng):
            LX = system.LX
            LY = system.LY
            obs = jnp.zeros((LY, LX), dtype=bool)
            obs_velocity = jnp.zeros((LY, LX, 2))
            rngs = jax.random.split(rng, 3)

            return {
                "cXs": system.cXs,
                "cYs": system.cYs,
                "cXYs": system.cXYs,
                "weights": system.weights,
                "obs": obs,
                "obs_velocity": obs_velocity,
                "N_new": jax.random.normal(rngs[0], (9, LY, LX)),
                "f_new": jax.random.normal(rngs[1], (9, LY, LX))
            }

        def step_fn(state):
            return compute_propagation(
                state["cXs"],
                state["cYs"],
                state["cXYs"],
                state["weights"],
                state["obs"],
                state["obs_velocity"],
                state["N_new"],
                state["f_new"]
            )

        test_utils.benchmark("benchmark compute propagation", init_fn, step_fn)


    def test_perf_compute_total_force(self):
        system = test_utils.load_test_config("params.ini")

        def init_fn(rng):
            LX = system.LX
            LY = system.LY
            rngs = jax.random.split(rng, 3)
            curvature_force = jax.random.normal(rngs[0], (LY, LX, 2))
            viscous_force = jax.random.normal(rngs[1], (LY, LX))
            rho = jax.random.normal(rngs[2], (LY, LX))

            return {
                "gravityX": system.gravityX,
                "gravityY": system.gravityY,
                "curvature_force": curvature_force,
                "viscous_force": viscous_force,
                "rho": rho
            }

        def step_fn(state):
            return compute_total_force(
                state["gravityX"],
                state["gravityY"],
                state["curvature_force"],
                state["viscous_force"],
                state["rho"]
            )

        test_utils.benchmark("benchmark compute total force", init_fn, step_fn)


    def test_perf_compute_density_velocity_pressure(self):
        system = test_utils.load_test_config("params.ini")

        def init_fn(rng):
            LX = system.LX
            LY = system.LY
            obs = jnp.zeros((LY, LX), dtype=bool)
            rngs = jax.random.split(rng, 7)

            return {
                "density_one": system.density_one,
                "density_two": system.density_two,
                "cXs": system.cXs,
                "cYs": system.cYs,
                "weights": system.weights,
                "phi_weights": system.phi_weights,
                "obs": obs,
                "pressure": jax.random.normal(rngs[0], (LY, LX)),
                "u": jax.random.normal(rngs[1], (LY, LX, 2)),
                "rho": jax.random.normal(rngs[2], (LY, LX)),
                "phase_field": jax.random.normal(rngs[3], (LY, LX)),
                "phi_grad": jax.random.normal(rngs[4], (LY, LX, 2)),
                "N": jax.random.normal(rngs[5], (9, LY, LX)),
                "total_force": jax.random.normal(rngs[6], (LY, LX, 2))
            }

        def step_fn(state):
            return compute_density_velocity_pressure(
                state["density_one"],
                state["density_two"],
                state["cXs"],
                state["cYs"],
                state["weights"],
                state["phi_weights"],
                state["obs"],
                state["pressure"],
                state["u"],
                state["rho"],
                state["phase_field"],
                state["phi_grad"],
                state["N"],
                state["total_force"]
            )

        test_utils.benchmark("benchmark compute density velocity pressure", init_fn, step_fn)


    def test_perf_compute_segregation(self):
        system = test_utils.load_test_config("params.ini")

        def init_fn(rng):
            LX = system.LX
            LY = system.LY
            rngs = jax.random.split(rng, 5)

            return {
                "width": system.width,
                "cXYs": system.cXYs,
                "weights": system.weights,
                "phi_weights": system.phi_weights,
                "phase_field": jax.random.normal(rngs[0], (LY, LX)),
                "phi_grad": jax.random.normal(rngs[1], (LY, LX, 2)),
                "pressure": jax.random.normal(rngs[2], (LY, LX)),
                "u": jax.random.normal(rngs[3], (LY, LX, 2)),
                "N_new": jax.random.normal(rngs[4], (9, LY, LX)),
            }

        def step_fn(state):
            return compute_segregation(
                state["width"],
                state["cXYs"],
                state["weights"],
                state["phi_weights"],
                state["phase_field"],
                state["phi_grad"],
                state["pressure"],
                state["u"],
                state["N_new"],
            )

        test_utils.benchmark("benchmark compute segregation", init_fn, step_fn)


if __name__ == "__main__":
    absltest.main()

