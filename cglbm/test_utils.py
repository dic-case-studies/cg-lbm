
import jax
from jax import vmap, jit
from jax import numpy as jnp
import time
from etils import epath

from cglbm.environment import System
import cglbm.config as cfg


def load_config(path: str) -> System:
    full_path = epath.resource_path("cglbm") / f'test-data/{path}'
    sys = cfg.load_config(full_path)

    return sys


# Note: We are assuming the function passed is already JITed
# Note: The result can fluxtuate if observing something < 50us
# Note: We could use python's timeit
def benchmark(
    name: str, init_fn, fn, iter: int = 100
) -> float:

    key = jax.random.PRNGKey(42)
    init_state = init_fn(key)

    times = []
    for i in range(iter):
        t = time.time_ns() / 1000
        fn(init_state).block_until_ready()
        times.append((time.time_ns() / 1000) - t)
    op_time = jnp.mean(jnp.array(times[1:]))
    op_time_std = jnp.std(jnp.array(times[1:]))

    print(f"""
    {name} jit time: {times[0] - op_time:.3f}us
    op time: {op_time:.3f} + {op_time_std:.3f} us,
    {jax.devices()[0].device_kind}
    """
    )

    return op_time

# TODO: Create one more benchmark for simulation
