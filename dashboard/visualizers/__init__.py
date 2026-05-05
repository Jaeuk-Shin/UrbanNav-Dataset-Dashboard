"""
Visualizer registry.

To add a new visualizer, create a module here and register its function in
``VISUALIZERS``.  Every visualizer has the signature::

    def vis_xxx(output: QueryOutput, root: Path, max_n: int) -> None
"""

from .image_grid import vis_image_grid
from .detection import vis_detection
from .mask import vis_mask
from .trajectory import vis_trajectory
from .table import vis_table
from .filter_summary import vis_filter_summary
from .filter_timeline import vis_filter_timeline

VISUALIZERS = {
    "image_grid": vis_image_grid,
    "detection": vis_detection,
    "mask": vis_mask,
    "trajectory": vis_trajectory,
    "table": vis_table,
    "filter_summary": vis_filter_summary,
    "filter_timeline": vis_filter_timeline,
}
