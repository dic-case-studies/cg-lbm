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


@jit
def compute_phase_field(f: jax.Array):
    """
    f: (k, LX, LY,)

    return: (LX,LY,)
    """
    return jnp.einsum("kij->ij", f)


@jit
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


@jit
@partial(vmap, in_axes=(None, None, None, 0, 0, 0, 1), out_axes=0)
@partial(vmap, in_axes=(None, None, None, 0, 0, 0, 1), out_axes=0)
def compute_mom(
    kin_visc_one: jnp.float32,
    kin_visc_two: jnp.float32,
    M_D2Q9: jax.Array,
    u: jax.Array,
    pressure: jax.Array,
    phase_field: jax.Array,
    N: jax.Array):
    """
    kin_visc_one: float32
    kin_visc_two: float32
    M_D2Q9: (k,k,)
    u: (X,Y,2,)
    pressure: (X,Y,)
    phase_field: (X,Y,)
    N: (k,X,Y)

    return ((X,Y,k), (X,Y,k), (X,Y))
    """
    alpha = 4.0 / 9.0
    u2 = u[0] * u[0] + u[1] * u[1]
    mom_eq = jnp.array([1.0,
                        -(2 + 18 * alpha) / 5.0 + 3.0 * (u2 + 2 * pressure),
                        (-7.0 + 27 * alpha) / 5 - 3.0 * (u2 + 3 * pressure),
                        u[0],
                        -u[0],
                        u[1],
                        -u[1],
                        u[0] * u[0] - u[1] * u[1],
                        u[0] * u[1]])

    inv_kin_visc = (phase_field / kin_visc_one) + \
        (1 - phase_field) / kin_visc_two
    kin_visc_local = inv_kin_visc

    mom = jnp.einsum('kl,l->k', M_D2Q9, N)

    return mom, mom_eq, kin_visc_local


@jit
@partial(vmap, in_axes=(None, None, None, None, 0, 0, 0, 0), out_axes=0)
@partial(vmap, in_axes=(None, None, None, None, 0, 0, 0, 0), out_axes=0)
def compute_viscosity_correction(
    invM_D2Q9,
    cMs,
    density_one,
    density_two,
    phi_grad,
    kin_visc_local,
    mom,
    mom_eq):

    tauL = 0.5 + 3 * kin_visc_local

    S_D2Q9 = jnp.ones(9)
    S_D2Q9 = S_D2Q9.at[7].set(1.0 / tauL)
    S_D2Q9 = S_D2Q9.at[8].set(1.0 / tauL)

    mom_diff = jnp.einsum(
        'kl,l,l->k', invM_D2Q9, S_D2Q9, (mom - mom_eq))

    viscous_force = -3.0 * kin_visc_local * \
        (density_one - density_two) * \
        jnp.einsum('kmn, k, n -> m',  cMs, mom_diff, phi_grad)

    return viscous_force