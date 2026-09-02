"""Search layer: Elasticsearch client, embedding model, three search modes.

Extracted from index.ipynb so the app and notebooks share one implementation.
"""

import os
from functools import lru_cache

from elasticsearch import Elasticsearch

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
INDEX_NAME = os.getenv("ES_INDEX", "hansard-chunks")
EMBED_MODEL = "multi-qa-MiniLM-L6-cos-v1"


@lru_cache(maxsize=1)
def get_es() -> Elasticsearch:
    return Elasticsearch(ES_URL)


@lru_cache(maxsize=1)
def get_model():
    # Imported lazily: keeps module import fast and torch out of processes that don't embed
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


def _filter_clauses(filters):
    return [{"term": {field: value}} for field, value in (filters or {}).items()]


def keyword_search(query, k=5, filters=None):
    body = {
        "size": k,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text^3", "debate_title", "speaker"],
                        "type": "best_fields",
                    }
                },
                "filter": _filter_clauses(filters),
            }
        },
    }
    hits = get_es().search(index=INDEX_NAME, body=body)["hits"]["hits"]
    return [h["_source"] | {"_score": h["_score"]} for h in hits]


def vector_search(query, k=5, filters=None):
    query_vector = get_model().encode(query, normalize_embeddings=True)
    knn = {
        "field": "text_vector",
        "query_vector": query_vector.tolist(),
        "k": k,
        "num_candidates": max(100, k * 10),
    }
    if filters:
        knn["filter"] = _filter_clauses(filters)
    hits = get_es().search(index=INDEX_NAME, knn=knn, size=k)["hits"]["hits"]
    return [h["_source"] | {"_score": h["_score"]} for h in hits]


def hybrid_search(query, k=5, filters=None, rrf_k=60):
    """Reciprocal Rank Fusion over the keyword and vector result lists."""
    keyword_results = keyword_search(query, k=k * 2, filters=filters)
    vector_results = vector_search(query, k=k * 2, filters=filters)

    scores, docs = {}, {}
    for results in (keyword_results, vector_results):
        for rank, doc in enumerate(results):
            doc_id = doc["chunk_id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rrf_k + rank + 1)
            docs[doc_id] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [docs[doc_id] | {"_rrf_score": score} for doc_id, score in ranked]


def find_debate(title_query):
    """Best-matching debate by title; returns (ext_id, title) or None."""
    body = {
        "size": 1,
        "query": {"match": {"debate_title": title_query}},
        "collapse": {"field": "debate_ext_id"},
    }
    hits = get_es().search(index=INDEX_NAME, body=body)["hits"]["hits"]
    if not hits:
        return None
    src = hits[0]["_source"]
    return src["debate_ext_id"], src["debate_title"]


def get_debate_chunks(debate_ext_id, max_chunks=200):
    body = {
        "size": max_chunks,
        "query": {"term": {"debate_ext_id": debate_ext_id}},
        "sort": [{"order_in_section": "asc"}, {"chunk_index": "asc"}],
    }
    hits = get_es().search(index=INDEX_NAME, body=body)["hits"]["hits"]
    return [h["_source"] for h in hits]
