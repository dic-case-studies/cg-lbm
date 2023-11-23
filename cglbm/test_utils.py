from cglbm.environment import System
import cglbm.config as cfg

from etils import epath


def load_config(path: str) -> System:
    full_path = epath.resource_path("cglbm") / f'test-data/{path}'
    sys = cfg.load_config(full_path)

    return sys
