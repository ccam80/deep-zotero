"""Search with automatic context expansion."""
from .models import StoredChunk, RetrievalResult
from .interfaces import VectorStoreProtocol


class Retriever:
    """
    Semantic search with context expansion.

    Wraps VectorStore to provide:
    - Automatic context expansion around hits
    - Result formatting with full metadata
    """

    def __init__(self, vector_store: VectorStoreProtocol):
        self.store = vector_store

    def search(
        self,
        query: str,
        top_k: int = 10,
        context_window: int = 1,
        filters: dict | None = None,
        where_document: dict | None = None,
    ) -> list[RetrievalResult]:
        """
        Search for relevant chunks and expand context.

        Args:
            query: Search query
            top_k: Number of results
            context_window: Chunks before/after to include (0-5)
            filters: Optional metadata filters
            where_document: Optional document-text constraint, applied inside
                the similarity query rather than to its results

        Returns:
            List of RetrievalResult with expanded context
        """
        return self.expand(
            self.store.search(
                query, top_k=top_k, filters=filters, where_document=where_document
            ),
            context_window=context_window,
        )

    def expand(
        self,
        hits: list[StoredChunk],
        context_window: int = 1,
    ) -> list[RetrievalResult]:
        """Turn raw chunk hits into results with surrounding context.

        Shared by semantic search and by lexical retrieval, which produces
        hits without a similarity score.
        """
        results = []
        for hit in hits:
            # Get adjacent chunks
            if context_window > 0:
                adjacent = self.store.get_adjacent_chunks(
                    hit.metadata["doc_id"],
                    hit.metadata["chunk_index"],
                    window=context_window
                )
            else:
                adjacent = []

            # Separate into before/after
            context_before = []
            context_after = []
            center_idx = hit.metadata["chunk_index"]

            for adj in adjacent:
                adj_idx = adj.metadata["chunk_index"]
                if adj_idx < center_idx:
                    context_before.append(adj.text)
                elif adj_idx > center_idx:
                    context_after.append(adj.text)

            # Handle journal_quartile - empty string from DB means None
            jq = hit.metadata.get("journal_quartile", "")
            journal_quartile = jq if jq else None

            results.append(RetrievalResult(
                chunk_id=hit.id,
                text=hit.text,
                score=hit.score,
                doc_id=hit.metadata["doc_id"],
                doc_title=hit.metadata["doc_title"],
                authors=hit.metadata["authors"],
                year=hit.metadata["year"] or None,
                page_num=hit.metadata["page_num"],
                chunk_index=hit.metadata["chunk_index"],
                citation_key=hit.metadata.get("citation_key", ""),
                publication=hit.metadata.get("publication", ""),
                tags=hit.metadata.get("tags", ""),
                collections=hit.metadata.get("collections", ""),
                section=hit.metadata.get("section", "unknown"),
                section_confidence=hit.metadata.get("section_confidence", 1.0),
                journal_quartile=journal_quartile,
                context_before=context_before,
                context_after=context_after,
                chunk_type=hit.metadata.get("chunk_type", "text"),
                metadata=hit.metadata,
            ))

        return results
