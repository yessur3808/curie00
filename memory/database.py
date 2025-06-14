# memory/database.py

import psycopg2
from psycopg2.extras import DictCursor
from pymongo import MongoClient
from .config import PG_CONN_INFO, MONGODB_URI, MONGODB_DB

# Postgres setup
def get_pg_conn():
    return psycopg2.connect(**PG_CONN_INFO, cursor_factory=DictCursor)

# MongoDB setup
mongo_client = MongoClient(MONGODB_URI)
mongo_db = mongo_client[MONGODB_DB]

# --- Database Initialization ---
def init_pg():
    with get_pg_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE EXTENSION IF NOT EXISTS "pgcrypto";
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                internal_id UUID UNIQUE DEFAULT gen_random_uuid(),
                telegram_id TEXT,
                slack_id TEXT,
                whatsapp_id TEXT,
                signal_id TEXT,
                phone_number TEXT,
                email TEXT,
                secret_username TEXT NOT NULL,
                is_master BOOLEAN NOT NULL DEFAULT FALSE,
                roles TEXT[] DEFAULT ARRAY[]::TEXT[],
                created_at TIMESTAMPTZ DEFAULT now(),
                updated_at TIMESTAMPTZ DEFAULT now(),
                updated_by TEXT NOT NULL
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversation_memory (
                id SERIAL PRIMARY KEY,
                user_internal_id UUID NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                FOREIGN KEY (user_internal_id) REFERENCES users(internal_id) ON DELETE CASCADE
            );
        """)
        conn.commit()

def init_mongo():
    mongo_db.research_memory.create_index([('topic', 1)])
    mongo_db.research_memory.create_index([('user_id', 1)])
    
    
def init_databases():
    init_pg()
    init_mongo()