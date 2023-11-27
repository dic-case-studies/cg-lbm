from functools import partial
import jax
from jax import vmap, jit, lax
from jax import numpy as jnp
import numpy as np


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
    """
    invM_D2Q9: (k,k,)
    cMs: (k,k,)
    density_one: jnp.float32
    density_two: jnp.float32
    phi_grad: (X,Y,2,)
    kin_visc_local: (X,Y,)
    mom: (X, Y, 9)
    mom_eq: (X, Y, 9)

    return (X, Y)
    """
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


@jit
@partial(vmap, in_axes=(None, 0, 0, 0, 0, 0, 0, 1), out_axes=1)
@partial(vmap, in_axes=(None, 0, 0, 0, 0, 0, 0, 1), out_axes=1)
def compute_collision(
    invM_D2Q9,
    obs,
    mom,
    mom_eq,
    kin_visc_local,
    interface_force,
    rho,
    N 
):
    """
    invM_D2Q9: (k,k,)
    obs: (X,Y)
    mom: (X, Y, 9)
    mom_eq: (X, Y, 9)
    kin_visc_local: (X,Y,)
    interface_force: (X,Y,2,)
    rho: (X,Y,)
    N: (k,X,Y)

    return (k,X, Y)
    """
    # TODO: For collision the u, pressure, rho are are supposed to be taken after
    # the compute_density_velocity_pressure
    tauL = 0.5 + 3 * kin_visc_local

    S_D2Q9 = jnp.ones(9)
    S_D2Q9 = S_D2Q9.at[7].set(1.0 / tauL)
    S_D2Q9 = S_D2Q9.at[8].set(1.0 / tauL)

    force = interface_force / rho

    force_eq = jnp.zeros(9)
    force_eq = force_eq.at[3].set(force[0])
    force_eq = force_eq.at[4].set(-force[0])
    force_eq = force_eq.at[5].set(force[1])
    force_eq = force_eq.at[6].set(-force[1])

    mom_diff = mom - S_D2Q9 * (mom - mom_eq) + (force_eq - 0.5 * S_D2Q9 * force_eq)
    N_new = jnp.einsum("kl,l->k", invM_D2Q9, mom_diff)
    
    # We will have to compute this to avoid divergence
    return lax.select(obs, N_new[np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])], N_new)
@partial(vmap, in_axes=(None, None, None, 0, 1, 0), out_axes=0)
@partial(vmap, in_axes=(None, None, None, 0, 1, 0), out_axes=0)
def surface_tension_force(
        surface_tension: jnp.float32,
        width: jnp.float32,
        weights: jax.Array,
        phase_field: jax.Array,
        dst_phase_field: jax.Array,
        phi_grad: jax.Array):
    """
    surface_tension: ()
    width: ()
    weights: (k,)
    phase_field: (X, Y,)
    dst_phase_field: (k, X, Y,)
    phi_grad: (X, Y, 2,)
    """
    phase_diff = dst_phase_field - phase_field
    laplacian_loc = 6 * jnp.einsum("k,k", phase_diff, weights)

    phase_term = (48 * phase_field * (1 - phase_field)
                  * (0.5 - phase_field)) / width
    phase_term -= (1.5 * width * laplacian_loc)

    curvature_force = surface_tension * phase_term * phi_grad

    return curvature_force
