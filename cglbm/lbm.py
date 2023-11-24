from functools import partial
import jax
from jax import vmap, jit
from jax import numpy as jnp


@jit
@partial(vmap, in_axes=(None, None, 0, 0), out_axes=1)
@partial(vmap, in_axes=(None, None, 0, 0), out_axes=1)
def eq_dist_phase_field(cXYs, weights, phi, u):
    """
    cXYs: (k, 2,)
    weights: (k,)
    phi: (LX, LY,)
    u: (LX, LY, 2,)

    return: (k, LX, LY,)
    """
    cu = jnp.sum(u * cXYs)
    u2 = jnp.sum(jnp.square(u))
    return weights * phi * (1.0 - 1.5 * u2 + 3.0 * cu + 4.5 * cu * cu)


@jit
def eq_dist(cXYs, weights, phi_weights, pressure, u):
    """
    cXYs: (k, 2,)
    weights: (k,)
    phi_weights: (k,)
    pressure: (1,)
    u: (2,)

    return: (k,)
    """
    cu = jnp.sum(u * cXYs)
    u2 = jnp.sum(jnp.square(u))
    
    neq_common_term = phi_weights + (weights + 3.0 * pressure) - (weights * u2 * 1.5)
    neq_k_zero_term = -3.0 * pressure
    neq_k_nonzero_term = weights * (3.0 * cu + 4.5 * cu * cu)
    
    k = jnp.arange(9)
    neq = neq_common_term + jnp.where(k == 0, neq_k_zero_term, neq_k_nonzero_term)
    
    return neq

_eq_dist = vmap(eq_dist, in_axes=(None, None, None, 0, 0), out_axes=1)
grid_eq_dist = jit(vmap(_eq_dist, in_axes=(None, None, None, 0, 0), out_axes=1))


def compute_phase_field(f: jax.Array):
    """
    f: (k, LX, LY,)

    return: (LX,LY,)
    """
    return jnp.einsum("kij->ij", f)


def compute_dst_phase_field(cXs: jax.Array, cYs: jax.Array, phase_field: jax.Array):
    """
    cXs: (k,)
    cYs: (k,)
    phase_field: (LX, LY,)

    return: (k, LX, LY,)
    """
    dst_phase_field = []

    for i, cx, cy in zip(jnp.arange(9), cXs, cYs):
        dst_phase_field.append(
            jnp.roll(phase_field, (-cx, -cy), axis=(0, 1)))

    return jnp.stack(dst_phase_field)


@jit
@partial(vmap, in_axes=(None, None, 1), out_axes=0)
@partial(vmap, in_axes=(None, None, 1), out_axes=0)
def compute_phi_grad(cXYs: jax.Array, weights: jax.Array, dst_phase_field: jax.Array):
    """
    cXYs: (k, 2,)
    weights: (k,)
    dst_phase_field: (k, LX, LY,)

    return: (LX, LY, 2)
    """
    phi_grad = 3 * jnp.einsum("k,k,kx->x", weights, dst_phase_field, cXYs)

    return phi_grad
