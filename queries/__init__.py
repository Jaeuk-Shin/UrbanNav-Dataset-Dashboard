"""
Query registry.

To add a new query, import its class here and append an instance to ``QUERIES``.
"""

from .overview import DatasetOverview
from .pedestrian import PedestrianCountQuery
from .object_presence import ObjectPresenceQuery
from .ego_velocity import EgoVelocityQuery
from .caption_search import CaptionSearchQuery

QUERIES: list = [
    DatasetOverview(),
    PedestrianCountQuery(),
    ObjectPresenceQuery(),
    EgoVelocityQuery(),
    CaptionSearchQuery(),
]
