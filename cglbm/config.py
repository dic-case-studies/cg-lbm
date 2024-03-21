import configparser
from cglbm.environment import System
from etils import epath
import jax.numpy as jnp

# TODO: This should throw an error if file not found


class SimulationParams:
    def __init__(self, config_file):
        self.parse_params(config_file)

    def parse_params(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        pt = config["Simulation"]

        # We could destructure into a map
        self.LX = int(pt.get("LX", 0))
        self.LY = int(pt.get("LY", 0))
        self.NL = 9
        self.nr_iterations = int(pt.get("nr_iterations", 0))
        self.nr_samples = int(pt.get("nr_samples", 0))
        self.kin_visc_one = float(pt.get("kin_visc_one", 0.0))
        self.kin_visc_two = float(pt.get("kin_visc_two", 0.0))
        self.density_one = float(pt.get("density_one", 0.0))
        self.density_two = float(pt.get("density_two", 0.0))
        self.gravityX = float(pt.get("gravityX", 0.0))
        self.gravityY = float(pt.get("gravityY", 0.0))
        self.width = float(pt.get("width", 0.0))
        self.surface_tension = float(pt.get("surface_tension", 0.0))
        self.ref_pressure = float(pt.get("ref_pressure", 0.0))
        self.uWallX = float(pt.get("uWallX", 0.0))
        self.drop_radius = float(pt.get("drop_radius", 0.0))
        self.contact_angle = float(pt.get("contact_angle", 45))
        self.Width = 4.0

    def print_config(self):
        print(f"LX = {self.LX} LY= {self.LY}")
        print(
            f"nr_iterations = {self.nr_iterations}  nr_samples = {self.nr_samples}")
        print(f"kin_viscosities = {self.kin_visc_one} {self.kin_visc_two}")
        print(f"density = {self.density_one} {self.density_two}")
        print(f"gravity = {self.gravityX} {self.gravityY}")
        print(f"interface width = {self.width}")
        print(f"surface tension = {self.surface_tension}")
        print(f"reference pressure = {self.ref_pressure}")
        print(f"wall velocity = {self.uWallX}")
        print(f"drop radius = {self.drop_radius}")
        print(f"contact angle = {self.contact_angle}")
        print(f"Width = {self.width}")


def load_config(config_file: str) -> System:
    """Loads a system from a given config file"""
    config = SimulationParams(config_file)

    from cglbm.d2q9 import NL, alpha, cXs, cYs, cXYs, cMs, weights, phi_weights, M_D2Q9, invM_D2Q9

    return System(
        LX=config.LX,
        LY=config.LY,
        NL=NL,
        kin_visc_one=config.kin_visc_one,
        kin_visc_two=config.kin_visc_two,
        density_one=config.density_one,
        density_two=config.density_two,
        gravityX=config.gravityX,
        gravityY=config.gravityY,
        width=config.width,
        surface_tension=config.surface_tension,
        ref_pressure=config.ref_pressure,
        # TODO: This has to be part of obstacle not be a part of config
        uWallX=config.uWallX,
        drop_radius=config.drop_radius,
        contact_angle=config.contact_angle,
        alpha=alpha,
        cXs=cXs,
        cYs=cYs,
        cXYs=cXYs,
        cMs=cMs,
        weights=weights,
        phi_weights=phi_weights,
        M_D2Q9=M_D2Q9,
        invM_D2Q9=invM_D2Q9
    )


def load_sandbox_config(path: str) -> System:
    full_path = epath.resource_path("cglbm") / f'sandbox-configs/{path}'
    sys = load_config(full_path)

    return sys
