CREATE TABLE IF NOT EXISTS conversation_memory (
    id SERIAL PRIMARY KEY,
    user_internal_id UUID NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    role TEXT NOT NULL,
    message TEXT NOT NULL,
    FOREIGN KEY (user_internal_id) REFERENCES users(internal_id) ON DELETE CASCADE
);