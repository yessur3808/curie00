# memory_manager.py

import os
import uuid
import psycopg2
from psycopg2.extras import DictCursor
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
import json

load_dotenv()

# --- Postgres setup ---
PG_CONN_INFO = {
    "host": os.getenv("POSTGRES_HOST"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "database": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
}

def get_pg_conn():
    return psycopg2.connect(**PG_CONN_INFO, cursor_factory=DictCursor)

# --- MongoDB setup ---
mongo_client = MongoClient(os.getenv("MONGODB_URI"))
mongo_db = mongo_client[os.getenv("MONGODB_DB")]

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

def init_memory():
    init_pg()
    init_mongo()

# --- User Management ---
def get_internal_id_by_secret_username(secret_username):
    """Get the internal_id (UUID) for a user by their secret_username."""
    with get_pg_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT internal_id FROM users WHERE secret_username ILIKE %s", (secret_username,))
        row = cur.fetchone()
        return str(row['internal_id']) if row else None
    
def get_or_create_user_internal_id(channel, external_id, secret_username=None, updated_by=None, is_master=False, roles=None):
    """
    Look up or create a user based on external channel ID.
    If creating, must supply secret_username and updated_by.
    """
    with get_pg_conn() as conn:
        cur = conn.cursor()
        field = f"{channel}_id"
        cur.execute(f"SELECT internal_id FROM users WHERE {field} = %s", (str(external_id),))
        row = cur.fetchone()
        if row:
            return str(row['internal_id'])
        # Create new user
        if not secret_username or not updated_by:
            raise ValueError("secret_username and updated_by are required to create a new user.")
        new_uuid = str(uuid.uuid4())
        cur.execute(
            f"""INSERT INTO users (internal_id, {field}, secret_username, updated_by, is_master, roles)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING internal_id""",
            (
                new_uuid,
                str(external_id),
                secret_username,
                updated_by,
                is_master,
                roles if roles else []
            )
        )
        conn.commit()
        return new_uuid

def set_user_roles(internal_id, roles, updated_by):
    with get_pg_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET roles = %s, updated_at = %s, updated_by = %s WHERE internal_id = %s",
            (roles, datetime.utcnow(), updated_by, str(internal_id))
        )
        conn.commit()

def set_user_master(internal_id, is_master=True, updated_by=None):
    with get_pg_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET is_master = %s, updated_at = %s, updated_by = %s WHERE internal_id = %s",
            (is_master, datetime.utcnow(), updated_by, str(internal_id))
        )
        conn.commit()

def get_user_by_internal_id(internal_id):
    with get_pg_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE internal_id = %s", (str(internal_id),))
        return cur.fetchone()

# --- Conversation (Postgres) ---
def save_conversation(user_internal_id, role, message):
    with get_pg_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO conversation_memory (user_internal_id, timestamp, role, message) VALUES (%s, %s, %s, %s)",
            (str(user_internal_id), datetime.utcnow(), role, message)
        )
        conn.commit()

def load_recent_conversation(user_internal_id, limit=10):
    with get_pg_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT role, message FROM conversation_memory WHERE user_internal_id = %s ORDER BY timestamp DESC LIMIT %s",
            (str(user_internal_id), limit)
        )
        results = cur.fetchall()
        return list(reversed([(row['role'], row['message']) for row in results]))

def clear_conversation(user_internal_id=None):
    with get_pg_conn() as conn:
        cur = conn.cursor()
        if user_internal_id:
            cur.execute("DELETE FROM conversation_memory WHERE user_internal_id = %s", (str(user_internal_id),))
        else:
            cur.execute("DELETE FROM conversation_memory")
        conn.commit()

# --- Research Memory (MongoDB) ---
def save_research(topic, content, user_internal_id=None):
    doc = {
        'topic': topic,
        'content': content,
        'timestamp': datetime.utcnow()
    }
    if user_internal_id is not None:
        doc['user_id'] = str(user_internal_id)
    mongo_db.research_memory.insert_one(doc)

def search_research(topic, user_internal_id=None):
    query = {'topic': topic}
    if user_internal_id is not None:
        query['user_id'] = str(user_internal_id)
    results = mongo_db.research_memory.find(query).sort('timestamp', -1)
    return [doc['content'] for doc in results]

def search_global_research(topic):
    return search_research(topic, user_internal_id=None)

# --- User Profile (MongoDB) ---
def get_user_profile(internal_id):
    """Returns the 'facts' dict for this user, or an empty dict if not found."""
    doc = mongo_db.user_profiles.find_one({"_id": str(internal_id)})
    return doc.get("facts", {}) if doc and "facts" in doc else {}

def update_user_profile(internal_id, new_facts: dict):
    """
    Adds/updates facts for the user in MongoDB.
    Merges with any existing facts.
    """
    if not isinstance(new_facts, dict):
        raise ValueError("new_facts must be a dict")
    update = {}
    for k, v in new_facts.items():
        update[f"facts.{k}"] = v
    mongo_db.user_profiles.update_one(
        {"_id": str(internal_id)},
        {
            "$set": update,
            "$currentDate": {"last_updated": True}
        },
        upsert=True
    )