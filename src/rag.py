"""Local RAG store backed by Chroma. Wraps LearningCards for retrieval."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from loguru import logger

from .listen import LearningCard


class CardStore:
    """Vector store of LearningCards. Designed to run fully offline."""

    def __init__(self, persist_dir: str | Path = "./artifacts/chroma") -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._collection = None

    def _ensure_collection(self):
        if self._collection is not None:
            return self._collection
        import chromadb  # noqa: PLC0415
        from chromadb.utils import embedding_functions  # noqa: PLC0415

        client = chromadb.PersistentClient(path=str(self.persist_dir))
        embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        self._collection = client.get_or_create_collection(
            name="linguaforge_cards",
            embedding_function=embedder,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    def add(self, cards: Iterable[LearningCard]) -> int:
        col = self._ensure_collection()
        cards = list(cards)
        if not cards:
            return 0
        ids = [c.card_id for c in cards]
        docs = [
            f"{c.native_text}\n{c.english_gloss}\n{c.cultural_note}".strip()
            for c in cards
        ]
        metas = [
            {
                "language_code": c.language_code,
                "card_type": c.card_type,
                "english_gloss": c.english_gloss,
                "tags": ",".join(c.tags),
            }
            for c in cards
        ]
        col.upsert(ids=ids, documents=docs, metadatas=metas)
        logger.success("Indexed {} cards.", len(cards))
        return len(cards)

    def query(
        self,
        text: str,
        *,
        language_code: str | None = None,
        card_type: str | None = None,
        n_results: int = 5,
    ) -> list[dict]:
        col = self._ensure_collection()
        where: dict = {}
        if language_code:
            where["language_code"] = language_code
        if card_type:
            where["card_type"] = card_type
        result = col.query(
            query_texts=[text],
            n_results=n_results,
            where=where or None,
        )
        out: list[dict] = []
        for i in range(len(result["ids"][0])):
            out.append(
                {
                    "id": result["ids"][0][i],
                    "document": result["documents"][0][i],
                    "metadata": result["metadatas"][0][i],
                    "distance": result["distances"][0][i] if "distances" in result else None,
                }
            )
        return out

    def count(self) -> int:
        col = self._ensure_collection()
        return col.count()
