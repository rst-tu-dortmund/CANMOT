"""Small set of OmegaConf resolvers required by CANMOT configs."""

from __future__ import annotations

import os
import math

from omegaconf import OmegaConf


def register_resolvers() -> None:
    resolvers = {
        "path_join": lambda *parts: os.path.join(*(str(part) for part in parts if part is not None)),
        "math": lambda operation, value: {
            "log": math.log,
            "exp": math.exp,
            "sqrt": math.sqrt,
        }[str(operation)](float(value)),
    }
    for name, resolver in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)
