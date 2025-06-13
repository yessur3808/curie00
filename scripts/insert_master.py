import uuid
import psycopg2
import os
import argparse
from psycopg2.extras import DictCursor
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

PG_CONN_INFO = {
    "host": os.getenv("POSTGRES_HOST"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "database": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
}

def get_pg_conn():
    return psycopg2.connect(**PG_CONN_INFO, cursor_factory=DictCursor)

def get_or_create_user_id(channel, external_id, name=None, email=None, is_master=False, internal_id=None, secret_username=None):
    if not secret_username:
        raise ValueError("secret_username is required!")

    with get_pg_conn() as conn:
        cur = conn.cursor()
        field = f"{channel}_id"
        
        # Check if user exists
        cur.execute(f"SELECT internal_id FROM users WHERE {field} = %s", (str(external_id),))
        row = cur.fetchone()
        if row:
            # Update the updated_at timestamp and updated_by
            cur.execute("""
                UPDATE users 
                SET updated_at = CURRENT_TIMESTAMP, 
                    updated_by = 'master'
                WHERE internal_id = %s
                RETURNING internal_id
            """, (str(row['internal_id']),))
            conn.commit()
            updated_row = cur.fetchone()
            return str(updated_row['internal_id'])

        # Use provided internal_id or generate new one
        new_uuid = internal_id if internal_id else str(uuid.uuid4())

        # Build insert statement with all required fields
        fields = [
            "internal_id", 
            field, 
            "is_master", 
            "secret_username", 
            "roles",
            "updated_by",
            "updated_at"  # Added this field
        ]
        values = [
            new_uuid, 
            str(external_id), 
            is_master, 
            secret_username,
            ["master"] if is_master else [],
            "master",
            'CURRENT_TIMESTAMP'  # Added this value
        ]
        placeholders = ["%s"] * (len(fields) - 1) + ["CURRENT_TIMESTAMP"]  # Special handling for timestamp

        # Add optional fields if provided
        if email:
            fields.append("email")
            values.append(email)
            placeholders.append("%s")

        sql = f"""
            INSERT INTO users ({', '.join(fields)}) 
            VALUES ({', '.join(placeholders)})
            RETURNING internal_id
        """
        cur.execute(sql, tuple(values[:-1]))  # Exclude the CURRENT_TIMESTAMP from values
        conn.commit()
        new_row = cur.fetchone()
        return str(new_row['internal_id'])

def delete_master_user():
    master_internal_id = os.getenv("MASTER_USER_ID")
    
    if not master_internal_id:
        raise RuntimeError("MASTER_USER_ID must be set in your .env file!")
    
    with get_pg_conn() as conn:
        cur = conn.cursor()
        try:
            # First, check if the master user exists
            cur.execute("""
                SELECT internal_id 
                FROM users 
                WHERE internal_id = %s AND is_master = TRUE
            """, (master_internal_id,))
            user = cur.fetchone()
            
            if not user:
                print("⚠️  Master user not found in database!")
                return False
            
            # Update the record first with final update timestamp
            cur.execute("""
                UPDATE users 
                SET updated_at = CURRENT_TIMESTAMP,
                    updated_by = 'master'
                WHERE internal_id = %s AND is_master = TRUE
            """, (master_internal_id,))
            
            # Then delete the master user
            cur.execute("""
                DELETE FROM users 
                WHERE internal_id = %s AND is_master = TRUE
            """, (master_internal_id,))
            conn.commit()
            
            if cur.rowcount > 0:
                print(f"🗑️  Successfully deleted master user with internal_id: {master_internal_id}")
                return True
            else:
                print("⚠️  Failed to delete master user!")
                return False
                
        except Exception as e:
            print(f"❌ Error deleting master user: {str(e)}")
            conn.rollback()
            return False

def ensure_master_user():
    master_channel = "telegram"
    master_external_id = os.getenv("MASTER_TELEGRAM_ID")
    master_name = os.getenv("MASTER_USER_NAME")
    master_email = os.getenv("MASTER_EMAIL", "")
    master_internal_id = os.getenv("MASTER_USER_ID")
    master_secret_username = os.getenv("MASTER_SECRET_USERNAME")
    
    if not master_internal_id:
        raise RuntimeError("MASTER_USER_ID must be set in your .env file!")
    
    if not master_external_id:
        raise RuntimeError("MASTER_TELEGRAM_ID must be set in your .env file!")
    
    if not master_secret_username:
        raise RuntimeError("MASTER_SECRET_USERNAME must be set in your .env file!")
    
    print(f"👤 Ensuring master user: {master_secret_username}")

    return get_or_create_user_id(
        channel=master_channel,
        external_id=master_external_id,
        name=master_name,
        email=master_email,
        is_master=True,
        internal_id=master_internal_id,
        secret_username=master_secret_username
    )


        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Manage master user in the database')
    parser.add_argument('-d', '--delete', action='store_true', help='Delete the master user')
    args = parser.parse_args()

    if args.delete:
        if delete_master_user():
            print("✅ Master user deletion completed!")
        else:
            print("❌ Master user deletion failed!")
    else:
        master_id = ensure_master_user()
        print(f"✅ Master user ensured in database with internal_id: {master_id}")