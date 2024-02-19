import jax.numpy as jnp
import orbax.checkpoint as ocp

from cglbm.environment import State

def validate_sim_params(nr_iterations: int, nr_snapshots: int, nr_checkpoints: int):
    """
    Args:
        nr_iterations: () Number of iterations
        nr_snapshots: () Number of snapshots
        nr_checkpoints: () Number of checkpoints
    Returns:
        None
    """
    assert nr_iterations % nr_snapshots == 0, "Number of iterations should be divisible by number of snapshots"
    snapshot_interval = nr_iterations // nr_snapshots

    assert nr_iterations % nr_checkpoints == 0, "Number of checkpoints should be divisible by number of snapshots"
    checkpoint_interval = nr_iterations // nr_checkpoints

    assert nr_checkpoints <= nr_snapshots, "Number of checkpoints must be less than or equal to snapshots"

    assert checkpoint_interval % snapshot_interval == 0, "If checkpoint interval is not multiple of snapshot interval, the program might not generate all the checkpoints"
    

def restore_state(mngr: ocp.CheckpointManager, dummy_state: dict | State = None):
    """
    Args:
        mngr: instance of orbax CheckpointManager
        dummy_state: (dict | State) (optional) holds the data structure of the state to be restored. If none, restored_state is a regular pytree
    Returns:
        restored_state: can be pytree or a state with data structre same as dummy_state
    """
    latest_step = mngr.latest_step()
    
    if dummy_state is not None:
        restored_state = mngr.restore(latest_step, args=ocp.args.StandardRestore(dummy_state))
    else:
        restored_state = mngr.restore(latest_step)
        
    return restored_state