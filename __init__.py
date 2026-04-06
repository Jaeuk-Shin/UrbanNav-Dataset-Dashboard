"""
OMR Dataset Explorer Dashboard
================================
Run:  streamlit run dashboard.py

Package layout::

    dashboard/
        types.py          Result dataclasses (FrameResult, SegmentResult, QueryOutput)
        query.py          Query abstract base class
        loaders.py        Cached data-loading helpers
        app.py            Streamlit main entry point
        queries/          One file per query; __init__ exports QUERIES list
        visualizers/      One file per viz type; __init__ exports VISUALIZERS dict

Extension:
    - New query:  add a file in queries/, import + append in queries/__init__.py
    - New viz:    add a file in visualizers/, import + register in visualizers/__init__.py
"""

from types import FrameResult, SegmentResult, QueryOutput
from query import Query
