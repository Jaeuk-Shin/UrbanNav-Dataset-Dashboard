"""
Query registry.

To add a new query, import its class here and append an instance to ``QUERIES``.
"""

from dash_queries.overview import DatasetOverview
from dash_queries.pedestrian import PedestrianCountQuery
from dash_queries.crosswalk import CrosswalkCountQuery
from dash_queries.object_presence import ObjectPresenceQuery
from dash_queries.ego_velocity import EgoVelocityQuery
from dash_queries.caption_search import CaptionSearchQuery
from dash_queries.curation_overview import CurationOverview
from dash_queries.filter_diagnostic import FilterDiagnostic
from dash_queries.filtered_browse import FilteredBrowse
from dash_queries.filter_examples import FilterExamples

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
