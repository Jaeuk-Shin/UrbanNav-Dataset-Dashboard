"""
Query registry.

To add a new query, import its class here and append an instance to ``QUERIES``.
"""

from dash_queries.overview import DatasetOverview
from dash_queries.pedestrian import PedestrianCountQuery
from dash_queries.object_presence import ObjectPresenceQuery
from dash_queries.ego_velocity import EgoVelocityQuery
from dash_queries.caption_search import CaptionSearchQuery

QUERIES: list = [
    DatasetOverview(),
    PedestrianCountQuery(),
    ObjectPresenceQuery(),
    EgoVelocityQuery(),
    CaptionSearchQuery(),
]
