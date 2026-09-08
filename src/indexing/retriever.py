"""In-Corpus fact and evidence retrieval engine."""

from typing import List, Optional
from src.models.fact import Fact
from src.indexing.store import KnowledgeStore


class InCorpusRetriever:
    """Retrieves facts and evidence strictly from the local in-corpus DuckDB knowledge store."""

    def __init__(self, store: KnowledgeStore):
        self.store = store

    def search_facts(self, query: str, limit: int = 10) -> List[Fact]:
        """Search facts by matching keywords against entity, metric, or verbatim quotes."""
        sql = """
            SELECT * FROM facts
            WHERE LOWER(metric) LIKE LOWER(?)
               OR LOWER(entity) LIKE LOWER(?)
               OR LOWER(verbatim_quote) LIKE LOWER(?)
            ORDER BY page_number
            LIMIT ?;
        """
        pattern = f"%{query}%"
        rows = self.store.conn.execute(sql, [pattern, pattern, pattern, limit]).fetchall()
        return self.store._rows_to_facts(rows)

    def get_document_facts(self, doc_id: str) -> List[Fact]:
        """Retrieve all facts extracted from a specific document."""
        sql = "SELECT * FROM facts WHERE doc_id = ? ORDER BY page_number;"
        rows = self.store.conn.execute(sql, [doc_id]).fetchall()
        return self.store._rows_to_facts(rows)
