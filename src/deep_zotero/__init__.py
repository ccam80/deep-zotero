"""DeepZotero."""
from importlib.metadata import version as _distribution_version

from .models import (
    ZoteroItem,
    PageExtraction,
    DocumentExtraction,
    ExtractedFigure,
    Chunk,
    StoredChunk,
    RetrievalResult,
    SearchResponse,
)

__version__ = _distribution_version("deep-zotero")

__all__ = [
    "__version__",
    "ZoteroItem",
    "PageExtraction",
    "DocumentExtraction",
    "ExtractedFigure",
    "Chunk",
    "StoredChunk",
    "RetrievalResult",
    "SearchResponse",
]
