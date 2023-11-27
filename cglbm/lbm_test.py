from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import cglbm.test_utils as test_utils
from cglbm.lbm import *


class LBMTest(absltest.TestCase):
    """Performance test for LBM Methods
    Note:These are for reference actual numbers will go down after fusion
    """

    def test_eq_dist_phase_field_perf(self):
        sys = test_utils.load_config("params.ini")

        def init_fn(rng):
            LX = sys.LX
            LY = sys.LY
            rng1, rng2 = jax.random.split(rng, 2)
            phi = jax.random.normal(rng1, (LX, LY))
            u = jax.random.normal(rng2, (LX, LY, 2))
            return {
                "phi": phi,
                "u": u
            }

        def step_fn(state):
            return eq_dist_phase_field(sys.cXYs, sys.weights, state["phi"], state["u"])

        test_utils.benchmark("benchmark eq dist phase field", init_fn, step_fn)

    
    def test_grid_eq_dist_perf(self):
        sys = test_utils.load_config("params.ini")

        def init_fn(rng):
            LX = sys.LX
            LY = sys.LY
            rng1, rng2 = jax.random.split(rng, 2)
            phi = jax.random.normal(rng1, (LX, LY))
            u = jax.random.normal(rng2, (LX, LY, 2))
            return {
                "phi": phi,
                "u": u
            }

        def step_fn(state):
            return grid_eq_dist(sys.cXYs, sys.weights, sys.phi_weights, state["phi"], state["u"])

        test_utils.benchmark("benchmark grid eq dist", init_fn, step_fn)

    
    def test_eq_dist_perf(self):
        sys = test_utils.load_config("params.ini")

        def init_fn(rng):
            LX = sys.LX
            LY = sys.LY
            rng1, rng2 = jax.random.split(rng, 2)
            phi = jax.random.normal(rng1, (1,))
            u = jax.random.normal(rng2, (2,))
            return {
                "phi": phi,
                "u": u
            }

        def step_fn(state):
            return eq_dist(sys.cXYs, sys.weights, sys.phi_weights, state["phi"], state["u"])

        test_utils.benchmark("benchmark eq dist", init_fn, step_fn)


    def test_compute_phase_field_perf(self):
        sys = test_utils.load_config("params.ini")

        def init_fn(rng):
            LX = sys.LX
            LY = sys.LY
            rng1, _ = jax.random.split(rng, 2)
            f = jax.random.normal(rng1, (9, LX, LY))
            return {
                "f": f
            }

        def step_fn(state):
            return compute_phase_field(state["f"])

        test_utils.benchmark("benchmark compute phase field", init_fn, step_fn)


    def test_compute_dst_phase_field_perf(self):
        sys = test_utils.load_config("params.ini")

        def init_fn(rng):
            LX = sys.LX
            LY = sys.LY
            rng1, _ = jax.random.split(rng, 2)
            phi = jax.random.normal(rng1, (LX, LY))
            return {
                "phi": phi
            }

        def step_fn(state):
            return compute_dst_phase_field(sys.cXs, sys.cYs, state["phi"])

        test_utils.benchmark("benchmark compute dst phase field", init_fn, step_fn)


    def test_compute_phi_grad_perf(self):
        sys = test_utils.load_config("params.ini")

        def init_fn(rng):
            LX = sys.LX
            LY = sys.LY
            rng1, _ = jax.random.split(rng, 2)
            dst_phase_field = jax.random.normal(rng1, (9, LX, LY))
            return {
                "dst_phase_field": dst_phase_field
            }

        def step_fn(state):
            return compute_phi_grad(sys.cXYs, sys.weights, state["dst_phase_field"])

        test_utils.benchmark("benchmark compute phi grad", init_fn, step_fn)


    def test_compute_mom_perf(self):
        sys = test_utils.load_config("params.ini")

        def init_fn(rng):
            LX = sys.LX
            LY = sys.LY
            rngs = jax.random.split(rng, 4)

            phase_field = jax.random.normal(rngs[0], (LX, LY))
            u = jax.random.normal(rngs[1], (LX, LY, 2))
            pressure = jax.random.normal(rngs[2], (LX, LY))
            N = jax.random.normal(rngs[3], (9, LX, LY))

            return {
                "kin_visc_one": sys.kin_visc_one,
                "kin_visc_two": sys.kin_visc_two,
                "M_D2Q9": sys.M_D2Q9,
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


    def test_compute_viscosity_correction_perf(self):
        sys = test_utils.load_config("params.ini")

        def init_fn(rng):
            LX = sys.LX
            LY = sys.LY
            rngs = jax.random.split(rng, 5)

            return {
                "invM_D2Q9": sys.invM_D2Q9,
                "cMs": sys.cMs,
                "density_one": sys.density_one,
                "density_two": sys.density_two,

                "phi_grad": jax.random.normal(rngs[0], (LX, LY, 2)),
                "kin_visc_local": jax.random.normal(rngs[1], (LX, LY)),
                "mom": jax.random.normal(rngs[2], (LX, LY, 9)),
                "mom_eq": jax.random.normal(rngs[3], (LX, LY, 9)),
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


    def test_compute_collision_perf(self):
        sys = test_utils.load_config("params.ini")

        def init_fn(rng):
            LX = sys.LX
            LY = sys.LY
            obs = jnp.zeros((LX, LY), dtype=bool)
            rngs = jax.random.split(rng, 7)

            return {
                "invM_D2Q9": sys.invM_D2Q9,
                "obs": obs,

                "mom": jax.random.normal(rngs[1], (LX, LY, 9)),
                "mom_eq": jax.random.normal(rngs[2], (LX, LY, 9)),
                "kin_visc_local": jax.random.normal(rngs[3], (LX, LY)),
                "interface_force": jax.random.normal(rngs[4], (LX, LY, 2)),
                "rho": jax.random.normal(rngs[5], (LX, LY)),
                "N": jax.random.normal(rngs[6], (9, LX, LY))
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

    def test_surface_tension_force_perf(self):
        sys = test_utils.load_config("params.ini")

        def init_fn(rng):
            LX = sys.LX
            LY = sys.LY
            rngs = jax.random.split(rng, 3)

            phase_field = jax.random.normal(rngs[0], (LX, LY))
            dst_phase_field = jax.random.normal(rngs[1], (9, LX, LY))
            phi_grad = jax.random.normal(rngs[2], (LX, LY, 2))

            return {
                "surface_tension": sys.surface_tension,
                "width": sys.width,
                "weights": sys.weights,
                "phase_field": phase_field,
                "dst_phase_field": dst_phase_field,
                "phi_grad": phi_grad,
            }

        def step_fn(state):
            return surface_tension_force(
                state["surface_tension"],
                state["width"],
                state["weights"],
                state["phase_field"],
                state["dst_phase_field"],
                state["phi_grad"]
                )

        test_utils.benchmark("benchmark surface tension force", init_fn, step_fn)

if __name__ == "__main__":
    absltest.main()

