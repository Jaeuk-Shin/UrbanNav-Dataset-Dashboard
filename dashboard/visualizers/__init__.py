"""
Visualizer registry.

To add a new visualizer, create a module here and register its function in
``VISUALIZERS``.  Every visualizer has the signature::

    def xxx(output: QueryOutput, root: Path, max_n: int) -> None
"""

from .image_grid import image_grid
from .detection import detection
from .mask import mask
from .trajectory import trajectory
from .table import table
from .filter_summary import filter_summary
from .filter_timeline import filter_timeline

VISUALIZERS = {
    "image_grid": image_grid,
    "detection": detection,
    "mask": mask,
    "trajectory": trajectory,
    "table": table,
    "filter_summary": filter_summary,
    "filter_timeline": filter_timeline,
}
