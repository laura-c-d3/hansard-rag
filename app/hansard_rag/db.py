"""Monitoring storage: log every conversation and feedback event to Postgres."""

import os
import uuid
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import RealDictCursor

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "dbname": os.getenv("POSTGRES_DB", "hansard"),
    "user": os.getenv("POSTGRES_USER", "hansard"),
    "password": os.getenv("POSTGRES_PASSWORD", "hansard"),
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    route TEXT NOT NULL,
    latency_s DOUBLE PRECISION,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    n_sources INTEGER
);

CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    conversation_id UUID REFERENCES conversations(id),
    ts TIMESTAMPTZ NOT NULL,
    thumbs_up BOOLEAN NOT NULL
);
"""


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def init_db():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(SCHEMA)


def log_conversation(question, result):
    conversation_id = str(uuid.uuid4())
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO conversations
               (id, ts, question, answer, route, latency_s, prompt_tokens, completion_tokens, n_sources)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                conversation_id,
                datetime.now(timezone.utc),
                question,
                result["answer"],
                result["route"],
                result["latency_s"],
                result["prompt_tokens"],
                result["completion_tokens"],
                len(result["sources"]),
            ),
        )
    return conversation_id


def log_feedback(conversation_id, thumbs_up):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO feedback (conversation_id, ts, thumbs_up) VALUES (%s, %s, %s)",
            (conversation_id, datetime.now(timezone.utc), thumbs_up),
        )
