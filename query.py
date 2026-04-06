"""Query abstract base class."""

from abc import ABC, abstractmethod
from pathlib import Path

from types import QueryOutput


class Query(ABC):
    """
    Base class for all dataset queries.

    To create a new query:
      1. Subclass Query in a new file under ``queries/``.
      2. Set ``name`` and ``description`` class attributes.
      3. Implement ``build_params`` (sidebar widgets) and ``execute`` (filtering logic).
      4. Import the class and append an instance to ``QUERIES`` in ``queries/__init__.py``.
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def build_params(self) -> dict:
        """Create Streamlit sidebar widgets and return the collected parameter dict."""
        ...

    @abstractmethod
    def execute(self, root: Path, segments: list, params: dict) -> QueryOutput:
        """Run the query over *segments* and return a :class:`QueryOutput`."""
        ...
