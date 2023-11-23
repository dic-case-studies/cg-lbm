from functools import partial
import jax
from jax import vmap, jit
from jax import numpy as jnp


@jit
@partial(vmap, in_axes=(None, None, 0, 0), out_axes=1)
@partial(vmap, in_axes=(None, None, 0, 0), out_axes=1)
def eq_dist_phase_field(cXYs, weights, phi, u):
    cu = jnp.sum(u * cXYs)
    u2 = jnp.sum(jnp.square(u))
    return weights * phi * (1.0 - 1.5 * u2 + 3.0 * cu + 4.5 * cu * cu)
