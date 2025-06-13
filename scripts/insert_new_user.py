import uuid

def get_or_create_user_id(channel, external_id):
    with get_pg_conn() as conn:
        cur = conn.cursor()
        # Try to find existing user
        cur.execute(f"SELECT internal_id FROM users WHERE {channel}_id = %s", (str(external_id),))
        row = cur.fetchone()
        if row:
            return row[0]
        # Create new user
        new_uuid = str(uuid.uuid4())
        cur.execute(f"INSERT INTO users (internal_id, {channel}_id) VALUES (%s, %s)", (new_uuid, str(external_id)))
        conn.commit()
        return new_uuid