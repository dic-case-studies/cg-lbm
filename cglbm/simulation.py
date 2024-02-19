from jax import jit, lax, tree_util
import numpy as np  # We need numpy here where we transfer data from the GPU
import orbax.checkpoint as ocp

from cglbm.environment import System, State
from cglbm.lbm import *
from cglbm.utils import restore_state, validate_sim_params

@jit
def simulation_step(system: System, state: State, idx: int) -> State:
    """
    Args:
        system: System
        state: State

    Returns:
        next_state: State
    """
    phase_field = compute_phase_field(state.f)
    dst_phase_field = compute_dst_phase_field(
        system.cXs, system.cYs, phase_field=phase_field)

    phi_grad = compute_phi_grad(system.cXYs, system.weights, dst_phase_field)

    curvature_force = compute_surface_tension_force(
        system.surface_tension,
        system.width,
        system.weights,
        phase_field,
        dst_phase_field,
        phi_grad)

    mom, mom_eq, kin_visc_local = compute_mom(
        system.kin_visc_one,
        system.kin_visc_two,
        system.M_D2Q9,
        state.u,
        state.pressure,
        phase_field,
        state.N)

    viscous_force = compute_viscosity_correction(
        system.invM_D2Q9,
        system.cMs,
        system.density_one,
        system.density_two,
        phi_grad,
        kin_visc_local,
        mom,
        mom_eq)

    total_force = compute_total_force(
        system.gravityX,
        system.gravityY,
        curvature_force,
        viscous_force,
        state.rho
    )

    rho, u, pressure, interface_force = compute_density_velocity_pressure(
        system.density_one,
        system.density_two,
        system.cXs,
        system.cYs,
        system.weights,
        system.phi_weights,
        state.obs,
        state.pressure,
        state.u,
        state.rho,
        phase_field,
        phi_grad,
        state.N,
        total_force
    )

    mom, mom_eq, kin_visc_local = compute_mom(
        system.kin_visc_one,
        system.kin_visc_two,
        system.M_D2Q9,
        u,
        pressure,
        phase_field,
        state.N)

    N_new = compute_collision(
        system.invM_D2Q9,
        state.obs,
        mom,
        mom_eq,
        kin_visc_local,
        interface_force,
        rho,
        state.N
    )

    f_new = compute_segregation(
        system.width,
        system.cXYs,
        system.weights,
        system.phi_weights,
        phase_field,
        phi_grad,
        pressure,
        u,
        N_new
    )

    N, f = compute_propagation(
        system.cXs,
        system.cYs,
        system.cXYs,
        system.weights,
        state.obs,
        state.obs_velocity,
        N_new,
        f_new
    )

    return state.replace(
        u=u,
        phase_field=phase_field,
        pressure=pressure,
        rho=rho,
        N=N,
        f=f
    )


@jit
def multi_step_simulation_block(system: System, state: State, nr_iter):
    return lax.fori_loop(0, nr_iter, lambda i, s: simulation_step(system, s, i), state)


# Note: There needs to be a separate function for calling
# multi_step_simulation_block so that we can shard and perform pmap later
def multi_step_simulation(system: System, state: State, nr_iterations: int, nr_snapshots: int = 10):
    # TODO: nr_iterations has to be divisible by nr_snapshots
    save_interval = nr_iterations // nr_snapshots

    results = [{
        "u": state["u"],
        "phase_field": state["phase_field"]
    }]

    for _ in range(nr_snapshots):
        state = multi_step_simulation_block(system, state, save_interval)
        results.append({
            "u": state["u"],
            "phase_field": state["phase_field"]
        })

    # Receive the buffer from the Accelerator
    results = jax.device_get(results)
    return tree_util.tree_map(
        lambda *rs: np.stack([np.array(r) for r in rs]),
        *results)


def multi_step_simulation_with_checkpointing(system: System, state: State, mngr: ocp.CheckpointManager, nr_iterations: int, nr_snapshots: int, nr_checkpoints: int, resume_from_last_checkpoint: bool = True):
    num_iter_completed = 0

    if resume_from_last_checkpoint and mngr.latest_step() is not None:
        num_iter_completed = mngr.latest_step()
        state = restore_state(mngr, state)
    else:
        mngr.save(0, args=ocp.args.StandardSave(state)) # creating checkpoint for initial state

    validate_sim_params(nr_iterations, nr_snapshots, nr_checkpoints)

    snapshot_interval = nr_iterations // nr_snapshots
    snapshot_iterations = range(num_iter_completed + snapshot_interval,
                                num_iter_completed + nr_iterations + 1, snapshot_interval)    

    results = [{
        "u": state["u"],
        "phase_field": state["phase_field"]
    }]

    for iteration_index in snapshot_iterations:
        state = multi_step_simulation_block(system, state, snapshot_interval)

        results.append({
            "u": state["u"],
            "phase_field": state["phase_field"]
        })

        mngr.save(iteration_index, args=ocp.args.StandardSave(state))

    results = jax.device_get(results)
    return tree_util.tree_map(
        lambda *rs: np.stack([np.array(r) for r in rs]),
        *results), state