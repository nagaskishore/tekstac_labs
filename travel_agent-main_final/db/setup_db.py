import sqlite3, os, sys

HERE = os.path.dirname(__file__)
DB_PATH = os.path.join(HERE, "travel_ai.sqlite")
SCHEMA_PATH = os.path.join(HERE, "schema.sql")
SEED_PATH = os.path.join(HERE, "seed_data.sql")

def setup_database(reset: bool = True) -> str:
    """Create SQLite DB, apply schema, seed data."""
    if reset and os.path.exists(DB_PATH):
        print(f"Removing existing DB: {DB_PATH}")
        os.remove(DB_PATH)

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON;")

        # Load schema
        if not os.path.exists(SCHEMA_PATH):
            raise FileNotFoundError(f"Schema file not found: {SCHEMA_PATH}")

        print("Applying schema...")
        with open(SCHEMA_PATH, "r") as f:
            schema = f.read()
        cur.executescript(schema)
        conn.commit()
        print("Schema applied.")

        # Seed data
        if os.path.exists(SEED_PATH):
            print("Seeding initial data...")
            with open(SEED_PATH, "r") as f:
                cur.executescript(f.read())
            conn.commit()
            print("Seed data loaded.")
        else:
            print("No seed_data.sql found; skipping seed.")

        print(f"Setup complete. DB ready at {DB_PATH}")
        return DB_PATH

    except Exception as e:
        print(f"? Error during setup: {e}")
        sys.exit(1)
    finally:
        if conn:
            conn.close()
    try:
        os.chmod(DB_PATH, 0o664)
    except Exception:
        pass

if __name__ == "__main__":
    setup_database()
