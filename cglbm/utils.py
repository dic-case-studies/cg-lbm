import jax.numpy as jnp
import orbax.checkpoint as ocp
from typing import Any, Tuple

from cglbm.environment import State, System


def validate_sim_params(nr_iterations: int, nr_snapshots: int, checkpoint_interval: int = None):
    """
    Args:
        nr_iterations: () Number of iterations
        nr_snapshots: () Number of snapshots
        checkpoint_interval: () (default: None) Timesteps between consecutive checkpoints
    Returns:
        None
    """
    assert nr_iterations % nr_snapshots == 0, "Number of iterations should be divisible by number of snapshots"

    if checkpoint_interval is not None:
        snapshot_interval = nr_iterations // nr_snapshots
        assert checkpoint_interval >= snapshot_interval, "Checkpoint interval must be greater than snapshot interval (nr_iterations // nr_snapshots)"
        assert checkpoint_interval % snapshot_interval == 0, "If checkpoint interval is not multiple of snapshot interval (nr_iterations // nr_snapshots), the program might not generate all the checkpoints"


def save_checkpoint(step: int, mngr: ocp.CheckpointManager, system: System, state: State):
    """
    Args:
        step: (int) time step at the checkpoint is to be saved
        mngr: instance of orbax CheckpointManager
        system: (System) instance of System to be saved 
        state: (State) instance of State to be saved
    Returns:
        bool indicating whether save was successful or not
    """
    checkpoint = {"system": system, "state": state}

    return mngr.save(step,
                     args=ocp.args.StandardSave(checkpoint))


def restore_checkpoint(mngr: ocp.CheckpointManager, dummy_system: System, dummy_state: State) -> Tuple[System, State]:
    """
    Args:
        mngr: instance of orbax CheckpointManager
        dummy_system: (System) holds the data structure of the system to be restored.
        dummy_state: (State) holds the data structure of the state to be restored.
    Returns:
        system: (System) instance of system restored from checkpoint
        state: (State) instance of state restored from checkpoint
    """
    latest_step = mngr.latest_step()
    assert latest_step is not None

    checkpoint = {"system": dummy_system, "state": dummy_state}

    checkpoint = mngr.restore(latest_step,
                              args=ocp.args.StandardRestore(checkpoint))

    return checkpoint["system"], checkpoint["state"]
