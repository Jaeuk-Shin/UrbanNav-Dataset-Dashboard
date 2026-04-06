"""
Visualizer registry.

To add a new visualizer, create a module here and register its function in
``VISUALIZERS``.  Every visualizer has the signature::

    def vis_xxx(output: QueryOutput, root: Path, max_n: int) -> None
"""

from visualizers.image_grid import vis_image_grid
from visualizers.detection import vis_detection
from visualizers.mask import vis_mask
from visualizers.trajectory import vis_trajectory
from visualizers.table import vis_table

VISUALIZERS = {
    "image_grid": vis_image_grid,
    "detection": vis_detection,
    "mask": vis_mask,
    "trajectory": vis_trajectory,
    "table": vis_table,
}
