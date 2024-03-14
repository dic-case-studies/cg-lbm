from functools import partial
import jax
from jax import vmap, jit, lax
from jax import numpy as jnp
from typing import Tuple

@jit
@partial(vmap, in_axes=(None, None, 0, 0), out_axes=1)
@partial(vmap, in_axes=(None, None, 0, 0), out_axes=1)
def eq_dist_phase_field(cXYs, weights, phi, u):
    """
    Args:
        cXYs: (k,v,)
        weights: (k,)
        phi: (i,j,)
        u: (i,j,v,)

    Returns:
        f: (k,i,j,)
    """
    cu = jnp.sum(u * cXYs)
    u2 = jnp.sum(jnp.square(u))
    return weights * phi * (1.0 - 1.5 * u2 + 3.0 * cu + 4.5 * cu * cu)


@jit
def eq_dist(cXYs, weights, phi_weights, pressure, u):
    """
    Args:
        cXYs: (k,v,)
        weights: (k,)
        phi_weights: (k,)
        pressure: ()
        u: (v,)

    Returns:
        neq: (k,)
    """
    cu = jnp.sum(u * cXYs)
    u2 = jnp.sum(jnp.square(u))

    neq_common_term = phi_weights + (weights * 3.0 * pressure) - (weights * u2 * 1.5)
    neq_k_zero_term = -3.0 * pressure
    neq_k_nonzero_term = weights * (3.0 * cu + 4.5 * cu * cu)

    k = jnp.arange(9)
    neq = neq_common_term + jnp.where(k == 0, neq_k_zero_term, neq_k_nonzero_term)

    return neq


_eq_dist = vmap(eq_dist, in_axes=(None, None, None, 0, 0), out_axes=1)
grid_eq_dist = jit(vmap(_eq_dist, in_axes=(None, None, None, 0, 0), out_axes=1))


#TODO: Add perf test
@jit
def compute_dst_obs(cXs: jax.Array, cYs: jax.Array, obs: jax.Array):
    """
    Args:
        cXs: (k,)
        cYs: (k,)
        obs: (i,j,)

    Returns:
        dst_obs: (k,i,j,)
    """
    dst_obs = []
    for i, cx, cy in zip(jnp.arange(9), cXs, cYs):
        dst_obs.append(
            jnp.roll(obs, (cx, cy), axis=(0, 1)))

    return jnp.stack(dst_obs)

#TODO: Add perf test
@jit
def compute_surface_normals(
    cXYs: jax.Array,
    weights: jax.Array,
    dst_obs: jax.Array,
    obs_indices: Tuple[jax.Array, jax.Array]):
    """
    Args:
        cXYs: (k,v,)
        weights: (k,)
        dst_obs: (k,i,j,)
        obs_indices: [(n,),(n,)]

    Returns:
        surface_normal: (n,2)
    """
    # TODO: Add "cs" (speed of sound in lattice units) in the System itself
    # TODO: the "cs_2" variable also needs to be part of compute_phi_grad where we hardcode it to 3
    cs = 1.0 / jnp.sqrt(3)
    cs_2 = 3

    grad_solid = cs_2 * jnp.einsum("k,kij,kv->ijv", weights, dst_obs, cXYs)[obs_indices]

    mag_grad_solid = jnp.sqrt(jnp.sum(jnp.square(grad_solid), axis=1))

    surface_normal = grad_solid / mag_grad_solid[:,jnp.newaxis]

    return surface_normal

    # fixed surface normals
    # return jnp.full((len(obs_indices[0]), 2), fill_value=jnp.array([-1, 0]))


@jit
def wetting_boundary_condition_solid(
    width: jnp.float32,
    contact_angle: jnp.float32,
    obs_indices: jax.Array,
    surface_normals: jax.Array,
    phase_field: jax.Array
):
    """
    Args:
        width: ()
        contact_angle: ()
        obs_indices: [(n,),(n,)]
        surface_normals: (n,2,)
        phase_field: (i,j,)
        
    Returns:
        phase_field: (i,j,)
    """
    epsilon = - 2 * jnp.cos(contact_angle) / width
    epsilon_inv = 1 / epsilon

    normal_indices = (jnp.array(obs_indices).T + surface_normals).astype(dtype=jnp.int32)
    phase_fluid = phase_field[tuple(normal_indices.T)]

    phase_fluid = jnp.abs(epsilon_inv * ((1 + epsilon) - jnp.sqrt((1 + epsilon)
                          ** 2 - 4 * epsilon * phase_fluid)) - phase_fluid)

    phase_field = phase_field.at[obs_indices].set(phase_fluid)

    return phase_field


@jit
def compute_phase_field(phase_field_old: jax.Array, f: jax.Array, obs_mask: jax.Array):
    """
    Args:
        phase_field_old: (i,j)
        f: (k,i,j,)
        obs_mask: (i,j)
    Returns:
        phase_field: (i,j,)
    """
    phase_field_new = jnp.einsum("kij->ij", f)

    phase_field = jnp.where(obs_mask, phase_field_old, phase_field_new)

    return phase_field
    


@jit
def compute_dst_phase_field(cXs: jax.Array, cYs: jax.Array, phase_field: jax.Array):
    """
    Args:
        cXs: (k,)
        cYs: (k,)
        phase_field: (i,j,)

    Returns:
        dst_phase_field: (k,i,j,)
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
    Args:
        cXYs: (k,v,)
        weights: (k,)
        dst_phase_field: (k,i,j,)

    Returns:
        phi_grad: (i,j,v,)
    """
    phi_grad = 3 * jnp.einsum("k,k,kv->v", weights, dst_phase_field, cXYs)

    return phi_grad


@jit
@partial(vmap, in_axes=(None, None, None, 0, 1, 0), out_axes=0)
@partial(vmap, in_axes=(None, None, None, 0, 1, 0), out_axes=0)
def compute_surface_tension_force(
    surface_tension: jnp.float32,
    width: jnp.float32,
    weights: jax.Array,
    phase_field: jax.Array,
    dst_phase_field: jax.Array,
    phi_grad: jax.Array
):
    """
    Args:
        surface_tension: ()
        width: ()
        weights: (k,)
        phase_field: (i,j,)
        dst_phase_field: (k,i,j,)
        phi_grad: (i,j,v,)

    Returns:
        curvature_force: (i,j,v,)
    """
    phase_diff = dst_phase_field - phase_field
    laplacian_loc = 6 * jnp.einsum("k,k", phase_diff, weights)

    phase_term = (48 * phase_field * (1 - phase_field)
                  * (0.5 - phase_field)) / width
    phase_term -= (1.5 * width * laplacian_loc)

    curvature_force = surface_tension * phase_term * phi_grad

    return curvature_force


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
    N: jax.Array
):
    """
    Args:
        kin_visc_one: ()
        kin_visc_two: ()
        M_D2Q9: (k,k,)
        u: (i,j,v,)
        pressure: (i,j,)
        phase_field: (i,j,)
        N: (k,i,j,)

    Returns:
        mom: (i,j,k)
        mom_eq: (i,j,k)
        kin_visc_local: (i,j,)
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
    kin_visc_local = 1 / inv_kin_visc

    mom = jnp.einsum('kl,l->k', M_D2Q9, N)

    return mom, mom_eq, kin_visc_local


@jit
@partial(vmap, in_axes=(None, None, None, None, 0, 0, 0, 0), out_axes=0)
@partial(vmap, in_axes=(None, None, None, None, 0, 0, 0, 0), out_axes=0)
def compute_viscosity_correction(
    invM_D2Q9: jax.Array,
    cMs: jax.Array,
    density_one: jnp.float32,
    density_two: jnp.float32,
    phi_grad: jax.Array,
    kin_visc_local: jax.Array,
    mom: jax.Array,
    mom_eq: jax.Array
):
    """
    Args:
        invM_D2Q9: (k,k,)
        cMs: (k,m,n,)
        density_one: ()
        density_two: ()
        phi_grad: (i,j,v,)
        kin_visc_local: (i,j,)
        mom: (i,j,k,)
        mom_eq: (i,j,k,)

    Returns:
        viscous_force: (i,j,v,)
    """
    tauL = 0.5 + 3 * kin_visc_local

    S_D2Q9 = jnp.ones(9)
    S_D2Q9 = S_D2Q9.at[7].set(1.0 / tauL)
    S_D2Q9 = S_D2Q9.at[8].set(1.0 / tauL)

    mom_diff = jnp.einsum(
        'kl,l,l->k', invM_D2Q9, S_D2Q9, (mom - mom_eq))

    viscous_force = -3.0 * kin_visc_local * \
        (density_one - density_two) * \
        jnp.einsum('kmn,k,n->m', cMs, mom_diff, phi_grad)

    return viscous_force


@jit
@partial(vmap, in_axes=(None, None, 0, 0, 0), out_axes=0)
@partial(vmap, in_axes=(None, None, 0, 0, 0), out_axes=0)
def compute_total_force(
    gravityX: jnp.float32,
    gravityY: jnp.float32,
    curvature_force: jax.Array,
    viscous_force: jax.Array,
    rho: jax.Array
):
    """
    Args:
        gravityX: ()
        gravityY: ()
        curvature_force: (i,j,v,)
        viscous_force: (i,j,v,)
        rho: (i,j,)

    Returns:
        total_force: (i,j,v,)
    """
    rest_force = jnp.stack([rho * gravityX, rho * gravityY])
    return rest_force + curvature_force + viscous_force


@jit
@partial(vmap, in_axes=(None, None, None, None, None,
         None, 0, 0, 0, 0, 0, 0, 1, 0), out_axes=0)
@partial(vmap, in_axes=(None, None, None, None, None,
         None, 0, 0, 0, 0, 0, 0, 1, 0), out_axes=0)
def compute_density_velocity_pressure(
    density_one: jnp.float32,
    density_two: jnp.float32,
    cXs: jax.Array,
    cYs: jax.Array,
    weights: jax.Array,
    phi_weights: jax.Array,
    obs: jax.Array,
    pressure_old: jax.Array,
    u_old: jax.Array,
    rho_old: jax.Array,
    phase_field: jax.Array,
    phi_grad: jax.Array,
    N: jax.Array,
    total_force: jax.Array
):
    """
    Args:
        density_one: ()
        density_two: ()
        cXs: (k,)
        cYs: (k,)
        weights: (k,)
        phi_weights: (k,)
        obs: (i,j,)
        pressure: (i,j,)
        u: (i,j,v,)
        rho: (i,j,)
        phase_field: (i,j,)
        phi_grad: (i,j,v,)
        N: (k,i,j,)
        total_force: (i,j,v,)

    Returns:
        rho: (i,j,)
        u: (i,j,v,)
        pressure: (i,j,)
        interface_force: (i,j,v,)
    """

    sumNX = jnp.dot(N, cXs)
    sumNY = jnp.dot(N, cYs)
    sumNV = jnp.stack([sumNX, sumNY])
    sumN = jnp.sum(N[1:])

    rho_new = density_one * phase_field + \
        density_two * (1 - phase_field)

    pressure_new = pressure_old

    for _ in range(10):
        u_new = sumNV + ((total_force -
                          pressure_new * (density_one - density_two) * phi_grad) * 0.5) / rho_new
        usq_new = jnp.sum(jnp.square(u_new))
        pressure_new = (sumN / 3.0 - weights[0] * usq_new * 0.5 -
                        (1 - phi_weights[0]) / 3.0) / (1 - weights[0])

    interface_force_new = total_force - pressure_new * \
        (density_one - density_two) * phi_grad

    rho = jnp.where(obs, rho_old, rho_new)
    u = jnp.where(obs, u_old, u_new)
    pressure = jnp.where(obs, pressure_old, pressure_new)
    interface_force = jnp.where(obs,
                                jnp.zeros(interface_force_new.shape),
                                interface_force_new)

    return rho, u, pressure, interface_force


@jit
@partial(vmap, in_axes=(None, 0, 0, 0, 0, 0, 0, 1), out_axes=1)
@partial(vmap, in_axes=(None, 0, 0, 0, 0, 0, 0, 1), out_axes=1)
def compute_collision(
    invM_D2Q9: jax.Array,
    obs: jax.Array,
    mom: jax.Array,
    mom_eq: jax.Array,
    kin_visc_local: jax.Array,
    interface_force: jax.Array,
    rho: jax.Array,
    N: jax.Array
):
    """
    Args:
        invM_D2Q9: (k,k,)
        obs: (i,j,)
        mom: (i,j,k,)
        mom_eq: (i,j,k,)
        kin_visc_local: (i,j,)
        interface_force: (i,j,v,)
        rho: (i,j,)
        N: (k,i,j,)

    Returns:
        N_new: (k,i,j,)
    """
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
    return lax.select(obs, N, N_new)


@jit
@partial(vmap, in_axes=(None, None, None, None, 0, 0, 0, 0, 1), out_axes=1)
@partial(vmap, in_axes=(None, None, None, None, 0, 0, 0, 0, 1), out_axes=1)
def compute_segregation(
    width: jnp.float32,
    cXYs: jax.Array,
    weights: jax.Array,
    phi_weights: jax.Array,
    phase_field: jax.Array,
    phi_grad: jax.Array,
    pressure: jax.Array,
    u: jax.Array,
    N_new: jax.Array
):
    """
    Args:
        width: ()
        cXYs: (k,v,)
        weights: (k,)
        phi_weights: (k,)
        phase_field: (i,j,)
        phi_grad: (i,j,v,)
        pressure: (i,j,)
        u: (i,j,v,)
        N_new: (k,i,j,)

    Returns:
        f_new: (k,i,j,)
    """
    phi_mag = jnp.sqrt(jnp.sum(jnp.square(phi_grad)))

    N_eq = eq_dist(cXYs, weights, phi_weights, pressure, u)

    phigrad_dot_c = jnp.sum(phi_grad * cXYs, axis=1)
    seg_term = (2.0 / width) * (1.0 - phase_field) * phase_field * \
        (phigrad_dot_c / phi_mag) * N_eq

    seg_term = jnp.where(phi_mag == 0.0, 0.0, seg_term)

    f_new = phase_field * N_new + seg_term
    return f_new


@jit
def compute_propagation(
    cXs: jax.Array,
    cYs: jax.Array,
    cXYs: jax.Array,
    weights: jax.Array,
    dst_obs: jax.Array,
    obsVel: jax.Array,
    N_new: jax.Array,
    f_new: jax.Array
):
    """
    Args:
        cXs: (k,)
        cYs: (k,)
        cXYs: (k,v,)
        weights: (k,)
        dst_obs: (k,i,j)
        obsVel: (i,j,v,)
        N_new: (k,i,j,)
        f_new: (k,i,j,)

    Returns:
        N: (k,i,j,)
        f: (k,i,j,)
    """
    # TODO: use state.N and state.f for boundary cases
    N_dst = []
    # TODO: f can be removed from propogation since phase_field calculation does not care about direction
    f_dst = []
    dst_obs_vel = []
    
    for i, cx, cy in zip(jnp.arange(9), cXs, cYs):
        N_dst.append(jnp.roll(N_new[i], (cx, cy), axis=(0, 1)))
        f_dst.append(jnp.roll(f_new[i], (cx, cy), axis=(0, 1)))
        dst_obs_vel.append(jnp.roll(obsVel, (cx, cy), axis=(0, 1)))

    N_dst = jnp.stack(N_dst)
    f_dst = jnp.stack(f_dst)
    dst_obs_vel = jnp.stack(dst_obs_vel)

    N_invert = N_new[jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])] + 6.0 * \
        jnp.einsum("k,kv,kijv->kij", weights, cXYs, dst_obs_vel)
    f_invert = f_new[jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])]

    N = jnp.where(dst_obs, N_invert, N_dst)
    f = jnp.where(dst_obs, f_invert, f_dst)

    return N, f
