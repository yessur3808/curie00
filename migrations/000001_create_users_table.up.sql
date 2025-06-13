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