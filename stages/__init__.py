"""Stage registry -- auto-discovers all BaseStage subclasses in this package."""

import importlib
import pkgutil

from stages.base import STAGES

# Auto-import all modules in this package so stages register via __init_subclass__
for _info in pkgutil.iter_modules(__path__):
    if _info.name != "base":
        importlib.import_module(f".{_info.name}", __name__)

__all__ = ["STAGES"]
