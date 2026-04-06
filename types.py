"""Result types returned by queries and consumed by visualizers."""

from dataclasses import dataclass, field


@dataclass
class FrameResult:
    """A single-frame hit."""
    segment: str
    frame_id: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class SegmentResult:
    """A whole-segment hit."""
    segment: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class QueryOutput:
    """Container returned by Query.execute()."""
    results: list          # list[FrameResult | SegmentResult]
    viz_type: str          # key into VISUALIZERS
    title: str
    description: str
