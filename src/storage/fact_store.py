from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

from src.models.core import Event


class ChromaFactStore:
    """Thin wrapper around ChromaDB for event and proposal context retrieval."""

    def __init__(self, persist_dir: str | Path = "data/chroma", collection_name: str = "facts") -> None:
        persist_path = Path(persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_path))
        self._collection: Collection = self._client.get_or_create_collection(name=collection_name)

    def add_event(self, event: Event) -> None:
        """Store one event as an embedded document."""
        document = f"{event.headline}\n\n{event.body}"
        self._collection.add(
            ids=[event.id],
            documents=[document],
            metadatas=[
                {
                    "source": event.source,
                    "url": event.url,
                    "timestamp": event.timestamp.isoformat(),
                    "tickers": ",".join(event.tickers),
                    "importance": event.importance_score,
                }
            ],
        )

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        """Store a generic document in the fact store."""
        self._collection.add(ids=[doc_id], documents=[text], metadatas=[metadata or {}])

    def search(
        self,
        query: str,
        max_results: int = 5,
        where: dict[str, str | int | float | bool] | None = None,
    ) -> list[dict[str, object]]:
        """Run semantic search and return normalized result items."""
        result = self._collection.query(query_texts=[query], n_results=max_results, where=where)

        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        normalized: list[dict[str, object]] = []
        for item_id, doc, meta, dist in zip(ids, docs, metas, dists):
            normalized.append(
                {
                    "id": item_id,
                    "document": doc,
                    "metadata": meta,
                    "distance": dist,
                }
            )
        return normalized

    def delete(self, doc_id: str) -> None:
        """Delete a document by id."""
        self._collection.delete(ids=[doc_id])
