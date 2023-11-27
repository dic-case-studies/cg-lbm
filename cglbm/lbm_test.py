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

if __name__ == "__main__":
    absltest.main()

