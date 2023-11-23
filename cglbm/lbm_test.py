from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import cglbm.test_utils as test_utils
from cglbm.lbm import eq_dist_phase_field


class LBMTest(absltest.TestCase):
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


if __name__ == "__main__":
    absltest.main()

