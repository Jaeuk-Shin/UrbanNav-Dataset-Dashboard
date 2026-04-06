"""
Query registry.

To add a new query, import its class here and append an instance to ``QUERIES``.
"""

from queries.overview import DatasetOverview
from queries.pedestrian import PedestrianCountQuery
from queries.object_presence import ObjectPresenceQuery
from queries.ego_velocity import EgoVelocityQuery
from queries.caption_search import CaptionSearchQuery

QUERIES: list = [
    DatasetOverview(),
    PedestrianCountQuery(),
    ObjectPresenceQuery(),
    EgoVelocityQuery(),
    CaptionSearchQuery(),
]
