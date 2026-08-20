"""
VectorSearchTool — wraps VectorStoreManager with a clean search interface.
"""
from typing import List, Dict, Any, Optional


class VectorSearchTool:
    def __init__(self, vector_store):
        self._vs = vector_store

    def search(
        self,
        query: str,
        n_results: int = 8,
        source_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        return self._vs.query(query, n_results=n_results, source_filter=source_filter)

    def search_per_source(self, query: str, n_per_source: int = 2) -> List[Dict[str, Any]]:
        return self._vs.query_per_source(query, n_per_source=n_per_source)

    def list_sources(self) -> List[str]:
        return self._vs.list_sources()

    def total_chunks(self) -> int:
        return self._vs.total_chunks()
