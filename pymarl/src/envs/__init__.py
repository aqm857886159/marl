from functools import partial
import sys
import os

try:
    from smac.env import MultiAgentEnv, StarCraft2Env
except ImportError:  # pragma: no cover - optional dependency
    from .multiagentenv import MultiAgentEnv

    StarCraft2Env = None

from .edge_marl_env import EdgeMARLEnv

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
if StarCraft2Env is not None:
    REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["edge_marl"] = partial(env_fn, env=EdgeMARLEnv)
REGISTRY["edge_marl"] = partial(env_fn, env=EdgeMARLEnv)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
