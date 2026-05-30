"""
Hybrid retriever: ChromaDB semantic search + BM25 → Reciprocal Rank Fusion → cross-encoder rerank.

Usage:
    from retriever import HybridRetriever, load_reranker

    reranker = load_reranker()   # shared across both retrievers

    blog_retriever = HybridRetriever("./chroma_db", "blog_tactics", reranker)
    monster_retriever = HybridRetriever("./chroma_db", "monsters", reranker)

    # At generation time:
    tactical_hits = blog_retriever.search("aquatic fire predator ambush", k=5)
    reference_hits = monster_retriever.search("shark fire creature", k=3,
                                               filters={"cr": "5"})

    for hit in tactical_hits:
        print(hit["metadata"]["title"])
        print(hit["content_text"])   # full text ready for prompt injection
"""

import re
import warnings
import os
from typing import Optional

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60   # standard constant; higher = smoother fusion, lower = rank-1 dominates


def load_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


def load_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).split()


class HybridRetriever:
    """
    One instance per ChromaDB collection. Shares the reranker across instances
    since it's the heaviest object (~25 MB).
    """

    def __init__(
        self,
        chroma_dir: str,
        collection_name: str,
        reranker: Optional[CrossEncoder] = None,
        embeddings: Optional[HuggingFaceEmbeddings] = None,
    ):
        self.collection_name = collection_name
        self.reranker = reranker
        self.available = False  # set True only if collection loads successfully

        self._embeddings = embeddings or HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        # Check collection exists before trying to connect
        import chromadb as _chromadb
        _client = _chromadb.PersistentClient(path=chroma_dir)
        existing = [c.name for c in _client.list_collections()]
        if collection_name not in existing:
            print(f"[{collection_name}] Collection not found — skipping (run embed script to build it)")
            self._ids = []
            self._docs = []
            self._metas = []
            self._id_to_idx = {}
            self._bm25 = BM25Okapi([[]])
            return

        print(f"[{collection_name}] Connecting to ChromaDB …")
        self._db = Chroma(
            collection_name=collection_name,
            embedding_function=self._embeddings,
            persist_directory=chroma_dir,
        )

        print(f"[{collection_name}] Building BM25 index …")
        raw = self._db._collection.get(include=["documents", "metadatas"])
        self._ids    = raw["ids"]
        self._docs   = raw["documents"]
        self._metas  = raw["metadatas"]
        self._id_to_idx = {doc_id: i for i, doc_id in enumerate(self._ids)}
        self._bm25 = BM25Okapi([_tokenize(d) for d in self._docs])
        self.available = True
        print(f"[{collection_name}] Ready — {len(self._ids)} documents indexed")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 5,
        candidate_k: int = 20,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        if not self.available:
            return []
        """
        Returns top-k results as list of dicts:
            {
                "page_content":  str,   # full search payload (title + tags + body)
                "content_text":  str,   # raw body text for prompt injection
                "metadata":      dict,  # slug, title, date, categories, tags, …
                "rrf_score":     float,
            }
        """
        candidates = self._rrf(query, candidate_k, filters)
        if self.reranker:
            candidates = self._rerank(query, candidates)
        return candidates[:k]

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _semantic_search(
        self, query: str, candidate_k: int, filters: Optional[dict]
    ) -> list[str]:
        """Returns ordered list of ChromaDB IDs from semantic search."""
        query_vec = self._embeddings.embed_query(query)
        kwargs = {"n_results": candidate_k, "include": ["distances"]}
        if filters:
            kwargs["where"] = filters
        result = self._db._collection.query(query_embeddings=[query_vec], **kwargs)
        return result["ids"][0]   # ordered best → worst

    def _bm25_search(self, query: str, candidate_k: int) -> list[str]:
        """Returns ordered list of ChromaDB IDs from BM25."""
        scores = self._bm25.get_scores(_tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:candidate_k]
        return [self._ids[i] for i in ranked_indices]

    def _rrf(
        self, query: str, candidate_k: int, filters: Optional[dict]
    ) -> list[dict]:
        """Fuse semantic + BM25 rankings via Reciprocal Rank Fusion."""
        sem_ids  = self._semantic_search(query, candidate_k, filters)
        bm25_ids = self._bm25_search(query, candidate_k)

        sem_rank  = {doc_id: rank for rank, doc_id in enumerate(sem_ids)}
        bm25_rank = {doc_id: rank for rank, doc_id in enumerate(bm25_ids)}

        all_ids = set(sem_rank) | set(bm25_rank)
        rrf_scores = {
            doc_id: (
                1.0 / (RRF_K + sem_rank.get(doc_id,  candidate_k)) +
                1.0 / (RRF_K + bm25_rank.get(doc_id, candidate_k))
            )
            for doc_id in all_ids
        }

        ranked = sorted(all_ids, key=lambda x: -rrf_scores[x])[:candidate_k]
        return [self._make_result(doc_id, rrf_scores[doc_id]) for doc_id in ranked
                if doc_id in self._id_to_idx]

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Re-score candidates with the cross-encoder and re-sort."""
        pairs  = [(query, c["page_content"]) for c in candidates]
        scores = self.reranker.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda x: -x["rerank_score"])

    def _make_result(self, doc_id: str, rrf_score: float) -> dict:
        idx  = self._id_to_idx[doc_id]
        meta = self._metas[idx]
        return {
            "page_content": self._docs[idx],
            "content_text": meta.get("content_text", self._docs[idx]),
            "metadata":     meta,
            "rrf_score":    rrf_score,
        }
