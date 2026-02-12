import sqlite3
import json
import pandas as pd
from datetime import date, datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

# Handle both relative and absolute imports
try:
    from api.datamodels import User, Trip, ChatHistory, TripPlanModel
except ImportError:
    try:
        from api.datamodels import User, Trip, ChatHistory, TripPlanModel
    except ImportError:
        from api.datamodels import User, Trip, ChatHistory, TripPlanModel

# Database path
DB_PATH = Path(__file__).parent / "travel_ai.sqlite"

print(DB_PATH)
def get_connection():
    """Get database connection with row factory"""
    conn = sqlite3.connect(DB_PATH,
timeout=30,                  # wait for up to 30s instead of immediate lock error
        check_same_thread=False,
        isolation_level=None 
)
    conn.row_factory = sqlite3.Row  # return dict-like rows
    return conn

# -------------------------
# UI Helper Functions (moved from app.py)
# -------------------------

def get_all_users() -> List[str]:
    """Get all user names for UI dropdown"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM users ORDER BY name")
        users = [row[0] for row in cursor.fetchall()]
        conn.close()
        return users
    except Exception as e:
        print(f"Error getting users: {e}")
        return []

def get_user_id_by_name(user_name: str) -> Optional[int]:
    """Get user ID from name"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE name = ?", (user_name,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    except Exception as e:
        print(f"Error getting user ID: {e}")
        return None

def get_user_name_by_id(user_id: int) -> str:
    """Get user name from ID"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM users WHERE id = ?", (user_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else "Unknown"
    except Exception as e:
        print(f"Error getting user name: {e}")
        return "Unknown"

def get_trips_by_user_name(user_name: str) -> List[Dict]:
    """Get all trips for a user by name"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT t.id, t.title, t.phase 
            FROM trips t 
            JOIN users u ON u.id = t.user_id 
            WHERE u.name = ?
            ORDER BY t.created_at DESC
        """, (user_name,))
        trips = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return trips
    except Exception as e:
        print(f"Error getting trips: {e}")
        return []

def load_table_as_dataframe(table_name: str) -> pd.DataFrame:
    """Load table data as DataFrame for UI display"""
    try:
        conn = sqlite3.connect(DB_PATH)  # Use regular connection for pandas
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        # If table has a created_at column, parse it as datetime
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        return df
    except Exception as e:
        print(f"Error loading table {table_name}: {e}")
        return pd.DataFrame()

def get_recent_chat_history(limit: int = 5) -> List[Dict]:
    """Get recent chat history for debugging"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role, content, phase, user_id,trip_id,created_at 
            FROM chat_history 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (limit,))
        chat = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return chat
    except Exception as e:
        print(f"Error getting recent chat: {e}")
        return []

# -------------------------
# User Context Helper
# -------------------------
def get_trip_context(user_id: int, trip_id: int, phase: str) -> dict:
    """
    Returns trip context for a given user, trip, and phase using the trip_summary view.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM trip_summary WHERE id=? AND user_name=(SELECT name FROM users WHERE id=?) AND phase=?",
        (trip_id, user_id, phase)
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else {}

# -------------------------
# Trips
# -------------------------
def start_new_trip(user_id: int, request: dict) -> Trip:
    """
    Create a new Trip based on user input.
    """
    trip = Trip(user_id=user_id, **request, trip_status="draft")
    trip_id = create_trip(trip)
    trip.id = trip_id
    return trip

# -------------------------
# Chat History
# -------------------------
def save_chat_message_service(msg: ChatHistory) -> ChatHistory:
    """
    Save a chat message and return it with ID.
    """
    cid = save_chat_message(msg)
    msg.id = cid
    return msg

def load_chat_history_service(trip_id: int) -> List[dict]:
    """
    Load chat history for a trip.
    Always return as list of dicts for UI compatibility.
    """
    return load_chat_history(trip_id)

# --------------------------
# Helper converters
# --------------------------
def _serialize_value(value):
    """Convert Python ? SQLite compatible"""
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (date, datetime)):
        return str(value)
    return value

def _deserialize_value(value, expected_type):
    """Convert SQLite ? Python type"""
    if value is None:
        return None
    if expected_type == bool:
        return bool(value)
    if expected_type in (dict, list):
        try:
            return json.loads(value)
        except Exception:
            return {}
    if expected_type == date:
        return datetime.fromisoformat(value).date()
    if expected_type == datetime:
        return datetime.fromisoformat(value)
    return value

# --------------------------
# USERS
# --------------------------
def create_user(user: User) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO users (name, email, profile, travel_preferences, travel_constraints)
        VALUES (?, ?, ?, ?, ?)
    """, (user.name, user.email, user.profile, user.travel_preferences, user.travel_constraints))
    conn.commit()
    uid = cur.lastrowid
    conn.close()
    return int(uid) if uid is not None else 0

def get_user_by_id(user_id: int) -> Optional[User]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id=?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return User(**dict(row))
    return None

# --------------------------
# TRIPS
# --------------------------
def create_trip(trip: Trip) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO trips (user_id, phase, title, origin, destination, trip_startdate, trip_enddate,
                           accommodation_type, no_of_adults, no_of_children, budget, currency, trip_status,
                           purpose, travel_preferences, travel_constraints)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        trip.user_id,
        trip.phase,
        trip.title,
        trip.origin,
        trip.destination,
        _serialize_value(trip.trip_startdate),
        _serialize_value(trip.trip_enddate),
        trip.accommodation_type,
        trip.no_of_adults,
        trip.no_of_children,
        trip.budget,
        trip.currency,
        trip.trip_status,
        trip.purpose,
        trip.travel_preferences,
        trip.travel_constraints
    ))
    conn.commit()
    tid = cur.lastrowid
    conn.close()
    return int(tid) if tid is not None else 0

def get_trip_by_id(trip_id: int) -> Optional[Trip]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM trips WHERE id=?", (trip_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return Trip(**dict(row))
    return None

def update_trip_status(trip_id: int, trip_status: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE trips SET trip_status=? WHERE id=?", (trip_status, trip_id))
    conn.commit()
    conn.close()

# --------------------------
# CHAT HISTORY
# --------------------------
def save_chat_message(msg: ChatHistory) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO chat_history (trip_id, user_id, role, phase, content, metadata, sequence_number, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        msg.trip_id,
        msg.user_id,
        msg.role,
        msg.phase,
        msg.content,
        msg.metadata,
        msg.sequence_number,
        _serialize_value(msg.created_at)
    ))
    conn.commit()
    cid = cur.lastrowid
    conn.close()
    return int(cid) if cid is not None else 0

def load_chat_history(trip_id: int) -> list[dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT role, content,user_id,trip_id FROM chat_history WHERE trip_id=? ORDER BY created_at", (trip_id,))
    rows = cur.fetchall()
    conn.close()
    # Always return as dicts for UI compatibility
    return [{"role": r[0], "content": r[1],"user_id":r[2],"trip_id":r[3]} for r in rows]

# --------------------------
# TRIP PLANS
# --------------------------
def create_trip_plan(trip_plan: TripPlanModel) -> int:
    """Create a new trip plan in the database"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO trip_plans (trip_id, itinerary_json, hotels_json, flights_json, daily_budget, total_estimated_cost, status, version, agent_metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        trip_plan.trip_id,
        trip_plan.itinerary_json,
        trip_plan.hotels_json,
        trip_plan.flights_json,
        trip_plan.daily_budget,
        trip_plan.total_estimated_cost,
        trip_plan.status,
        trip_plan.version,
        trip_plan.agent_metadata
    ))
    conn.commit()
    plan_id = cur.lastrowid
    conn.close()
    return int(plan_id) if plan_id is not None else 0

def get_trip_plan_by_trip_id(trip_id: int, version: Optional[int] = None) -> Optional[TripPlanModel]:
    """Get trip plan by trip ID. If version not specified, gets latest version."""
    conn = get_connection()
    cur = conn.cursor()
    
    if version is not None:
        cur.execute("SELECT * FROM trip_plans WHERE trip_id=? AND version=?", (trip_id, version))
    else:
        cur.execute("""
            SELECT * FROM trip_plans 
            WHERE trip_id=? 
            ORDER BY version DESC 
            LIMIT 1
        """, (trip_id,))
    
    row = cur.fetchone()
    conn.close()
    
    if row:
        return TripPlanModel(**dict(row))
    return None

def get_all_trip_plan_versions(trip_id: int) -> List[TripPlanModel]:
    """Get all versions of trip plans for a trip"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM trip_plans 
        WHERE trip_id=? 
        ORDER BY version DESC
    """, (trip_id,))
    rows = cur.fetchall()
    conn.close()
    
    return [TripPlanModel(**dict(row)) for row in rows]

def update_trip_plan_status(plan_id: int, status: str) -> bool:
    """Update trip plan status (draft, approved, rejected)"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("UPDATE trip_plans SET status=? WHERE id=?", (status, plan_id))
        conn.commit()
        updated = cur.rowcount > 0
        conn.close()
        return updated
    except Exception as e:
        print(f"Error updating trip plan status: {e}")
        return False

def delete_trip_plan(plan_id: int) -> bool:
    """Delete a trip plan"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM trip_plans WHERE id=?", (plan_id,))
        conn.commit()
        deleted = cur.rowcount > 0
        conn.close()
        return deleted
    except Exception as e:
        print(f"Error deleting trip plan: {e}")
        return False

def save_travel_plan_to_db(travel_plan, trip_id: int, version: int = 1) -> int:
    """Helper function to save TravelPlan object to database"""
    trip_plan_model = TripPlanModel.from_travel_plan(travel_plan, trip_id, version)
    return create_trip_plan(trip_plan_model)

def get_trip_with_plan(trip_id: int) -> Optional[Dict]:
    """Get trip with its latest plan for UI display"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                t.*,
                tp.itinerary_json,
                tp.hotels_json,
                tp.flights_json,
                tp.daily_budget as plan_daily_budget,
                
                tp.total_estimated_cost,
                tp.status as plan_status,
                tp.generated_at as plan_generated_at,
                tp.version as plan_version
            FROM trips t
            LEFT JOIN trip_plans tp ON t.id = tp.trip_id 
                AND tp.version = (SELECT MAX(version) FROM trip_plans WHERE trip_id = t.id)
            WHERE t.id = ?
        """, (trip_id,))
        row = cur.fetchone()
        conn.close()
        
        return dict(row) if row else None
    except Exception as e:
        print(f"Error getting trip with plan: {e}")
        return None
    
#print(get_all_trip_plan_versions(1))

#-----
#added by me
##----

def get_last_user_messages_pre_trip(user_id: int, phase: str, limit: int = 2) -> List[Dict]:
    """
    Fetch the last N USER messages for a fresh trip context (trip_id = 0)
    for the same user and phase. Returns messages in chronological order.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role, content, phase, user_id, trip_id, created_at
            FROM chat_history
            WHERE user_id = ?
              AND phase = ?
              AND trip_id = 0
              AND role = 'user'
            ORDER BY created_at DESC
            LIMIT ?
        """, (user_id, phase, limit))
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return list(reversed(rows))  # chronological (oldest -> newest)
    except Exception as e:
        print(f"Error get_last_user_messages_pre_trip: {e}")
        return []
    
def attach_pre_trip_chat_to_trip(user_id: int, phase: str, new_trip_id: int, limit: int = 20) -> int:
    """
    Attach last N chat_history rows (trip_id=0) to the newly created trip_id.
    Returns number of rows updated.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
            UPDATE chat_history
            SET trip_id = ?
            WHERE id IN (
                SELECT id
                FROM chat_history
                WHERE user_id = ?
                  AND phase = ?
                  AND trip_id = 0
                ORDER BY created_at DESC
                LIMIT ?
            )
        """, (new_trip_id, user_id, phase, limit))

        conn.commit()
        updated = cur.rowcount
        conn.close()
        return updated
    except Exception as e:
        print(f"Error attach_pre_trip_chat_to_trip: {e}")
        return 0
def get_latest_thread_id_for_trip(trip_id: int) -> Optional[str]:
    """
    Get thread_id stored in agent_metadata for the latest trip plan.
    Returns None if not found.
    """
    try:
        plan = get_trip_plan_by_trip_id(trip_id)  # latest
        if not plan or not plan.agent_metadata:
            return None
        meta = json.loads(plan.agent_metadata)
        return meta.get("thread_id")
    except Exception as e:
        print(f"Error get_latest_thread_id_for_trip: {e}")
        return None
    

if __name__=='__main__':
    print(get_trip_with_plan(23))