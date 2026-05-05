"""
Query registry.

To add a new query, import its class here and append an instance to ``QUERIES``.
"""

from .overview import DatasetOverview
from .pedestrian import PedestrianCountQuery
from .crosswalk import CrosswalkCountQuery
from .object_presence import ObjectPresenceQuery
from .ego_velocity import EgoVelocityQuery
from .caption_search import CaptionSearchQuery
from .curation_overview import CurationOverview
from .filter_diagnostic import FilterDiagnostic
from .filtered_browse import FilteredBrowse
from .filter_examples import FilterExamples

QUERIES: list = [
    DatasetOverview(),
    PedestrianCountQuery(),
    CrosswalkCountQuery(),
    ObjectPresenceQuery(),
    EgoVelocityQuery(),
    CaptionSearchQuery(),
    CurationOverview(),
    FilterDiagnostic(),
    FilteredBrowse(),
    FilterExamples(),
]
