"""
Component registry — the interchange mechanism.

Every swappable part of the environment (an observation term, a reward term, a
termination check, an action mapping, a command sampler) registers itself here
under a string name.  The config (config.py) then selects parts *by name*, and
the environment (env/mjx_env.py) looks them up.  Nothing in the env hard-codes a
specific reward or observation — to swap a part you write a new function, decorate
it with the right registry, and change one name in the config.

    @REWARDS.register("track_lin_vel")
    def _(env, data, action, info): ...

    REWARDS.get("track_lin_vel")        # -> the function
    REWARDS.names()                     # -> ["track_lin_vel", ...]

Component call signatures (kept deliberately uniform so parts are interchangeable):

    observation(env, data, info)        -> 1-D array        (concatenated by env)
    reward_term(env, data, action, info)-> scalar           (weighted+summed by env)
    termination(env, data, info)        -> bool scalar       (OR-ed by env)
    action_map(env, action, data, info) -> ctrl vector (nu,)
    command_sample(rng, cfg)            -> command vector

`env` is the MjxEnv instance, passed read-only so components can reach model facts
(nominal pose, ctrl ranges, dt, body ids) without global state.
"""

from __future__ import annotations

from typing import Callable, Dict, List


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, Callable] = {}

    def register(self, name: str) -> Callable[[Callable], Callable]:
        def deco(fn: Callable) -> Callable:
            if name in self._items:
                raise ValueError(f"{self.kind} '{name}' already registered")
            self._items[name] = fn
            return fn
        return deco

    def get(self, name: str) -> Callable:
        if name not in self._items:
            raise KeyError(
                f"unknown {self.kind} '{name}'. registered: {self.names()}"
            )
        return self._items[name]

    def names(self) -> List[str]:
        return sorted(self._items)


# The five interchange points of the environment.
OBSERVATIONS = Registry("observation")
REWARDS      = Registry("reward")
TERMINATIONS = Registry("termination")
ACTIONS      = Registry("action")
COMMANDS     = Registry("command")
