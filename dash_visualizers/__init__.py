"""
Visualizer registry.

To add a new visualizer, create a module here and register its function in
``VISUALIZERS``.  Every visualizer has the signature::

    def vis_xxx(output: QueryOutput, root: Path, max_n: int) -> None
"""

from dash_visualizers.image_grid import vis_image_grid
from dash_visualizers.detection import vis_detection
from dash_visualizers.mask import vis_mask
from dash_visualizers.trajectory import vis_trajectory
from dash_visualizers.table import vis_table
from dash_visualizers.filter_summary import vis_filter_summary
from dash_visualizers.filter_timeline import vis_filter_timeline

VISUALIZERS = {
    "image_grid": vis_image_grid,
    "detection": vis_detection,
    "mask": vis_mask,
    "trajectory": vis_trajectory,
    "table": vis_table,
    "filter_summary": vis_filter_summary,
    "filter_timeline": vis_filter_timeline,
}
