
import jax
import configparser
from typing import Tuple

from chex import dataclass

from cglbm.base import Base

@dataclass
class State(Base):
    """Dynamic tensors that change with each iteration of the simulation.

    Attributes:
        rho: (LX, LY,) Density
        pressure: (LX, LY,) Pressure
        u: (LX, LY, x,) Velocity vector
        phase_field: (LX, LY,) Phase field
        obs: (LX, LY,) Obstacle
        obs_velocity: (LX, LY, x) Velocity vector of obstacle
        obs_indices: Tuple(x_cords, ycords) Tuple of X and Y coordinates of the obstacles
        f: (k, LX, LY) Phase profile
        N: (k, LX, LY) 
    """
    rho: jax.Array
    pressure: jax.Array
    u: jax.Array
    phase_field: jax.Array
    # TODO: Obs and obs_velocity can be part of a separate class
    obs: jax.Array
    obs_velocity: jax.Array
    obs_indices: Tuple[jax.Array, jax.Array]

    f: jax.Array
    N: jax.Array

# TODO: This class has to change to accomodate ghost nodes when distributing across devices
# @dataclass
# class Boundary():
#     obs_mask: jax.Array
#     obs_velocity: jax.Array
#     obs_indices: jax.Array

@dataclass
class System(Base):
    """Descibes a physical environment

    Attributes:
        LX: int
        LY: int
        NL: int

        kin_visc_one: float
        kin_visc_two: float
        density_one: float
        density_two: float
        gravityX: float
        gravityY: float
        width: int
        surface_tension: float
        ref_pressure: float
        uWallX: float
        drop_radius: float
        alpha: float

        # N-D Constants
        # [k]
        cXs: jax.Array
        cYs: jax.Array
        # TODO: check if cXYs is needed
        cXYs: jax.Array
        cMs: jax.Array
        weights: jax.Array
        phi_weights: jax.Array
        M_D2Q9: jax.Array
        invM_D2Q9: jax.Array
    """
    LX: int
    LY: int
    NL: int

    kin_visc_one: float
    kin_visc_two: float
    density_one: float
    density_two: float
    gravityX: float
    gravityY: float
    width: int
    surface_tension: float
    ref_pressure: float
    uWallX: float
    drop_radius: float
    alpha: float
    contact_angle: float

    # N-D Constants
    # [k]
    cXs: jax.Array
    cYs: jax.Array
    # TODO: check if cXYs is needed
    cXYs: jax.Array
    cMs: jax.Array
    weights: jax.Array
    phi_weights: jax.Array
    M_D2Q9: jax.Array
    invM_D2Q9: jax.Array
