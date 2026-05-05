"""
Query registry.

To add a new query, import its class here and append an instance to ``QUERIES``.
"""

from .overview import DatasetOverview
from .pedestrian import PedestrianCount
from .crosswalk import CrosswalkCount
from .object_presence import ObjectPresence
from .ego_velocity import EgoVelocity
from .caption_search import CaptionSearch
from .curation_overview import CurationOverview
from .filter_diagnostic import FilterDiagnostic
from .filtered_browse import FilteredBrowse
from .filter_examples import FilterExamples

QUERIES: list = [
    DatasetOverview(),
    PedestrianCount(),
    CrosswalkCount(),
    ObjectPresence(),
    EgoVelocity(),
    CaptionSearch(),
    CurationOverview(),
    FilterDiagnostic(),
    FilteredBrowse(),
    FilterExamples(),
]
