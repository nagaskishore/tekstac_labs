-- TravelMate AI - Complete Database Schema
-- Production-ready schema for 4-phase travel AI system

-- USERS TABLE
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL CHECK(email LIKE '%_@_%._%'),
    profile TEXT DEFAULT NULL,
    travel_preferences TEXT DEFAULT NULL,
    travel_constraints TEXT DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- TRIPS TABLE  
CREATE TABLE trips (
    -- System identifiers
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    
    -- Mandatory trip definition
    phase TEXT NOT NULL CHECK(phase IN ('phase1_langflow', 'phase2_crewai', 'phase3_autogen', 'phase4_langgraph')),
    title TEXT NOT NULL DEFAULT 'My Trip',
    origin TEXT NOT NULL,
    destination TEXT NOT NULL,
    trip_startdate DATE NOT NULL,
    trip_enddate DATE NOT NULL,
    accommodation_type TEXT NOT NULL CHECK(accommodation_type IN ('hotel', 'resort', 'hostel', 'apartment', 'guesthouse', 'luxury', 'own_place', 'friend_place', 'official_accommodation', 'budget', 'family-friendly', 'business', 'youth hostel')),
    no_of_adults INTEGER NOT NULL DEFAULT 1,
    no_of_children INTEGER NOT NULL DEFAULT 0,
    
    -- Mandatory with defaults
    budget REAL NOT NULL DEFAULT 500,
    currency TEXT NOT NULL DEFAULT 'USD',
    trip_status TEXT NOT NULL DEFAULT 'draft' CHECK(trip_status IN ('draft', 'confirmed', 'in_progress', 'completed', 'cancelled')),
    
    -- Optional fields with defaults
    purpose TEXT DEFAULT 'none',
    travel_preferences TEXT DEFAULT 'none',
    travel_constraints TEXT DEFAULT 'none',
    
    -- System metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    FOREIGN KEY(user_id) REFERENCES users(id),
    CHECK(no_of_adults >= 1),
    CHECK(no_of_children >= 0),
    CHECK(budget >= 0),
    CHECK(trip_enddate > trip_startdate),
    CHECK(julianday(trip_enddate) - julianday(trip_startdate) >= 1)
);

-- TRIP PLANS TABLE (stores AI-generated trip plan results)
CREATE TABLE trip_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trip_id INTEGER NOT NULL,
    itinerary_json TEXT, -- Itinerary data as JSON
    hotels_json TEXT,    -- Hotel suggestions as JSON  
    flights_json TEXT,   -- Flight suggestions as JSON
    daily_budget REAL DEFAULT 0.0,
    total_estimated_cost REAL DEFAULT 0.0,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'draft' CHECK(status IN ('draft', 'approved', 'rejected')),
    version INTEGER DEFAULT 1, -- Allow multiple plan versions
    agent_metadata TEXT, -- JSON for agent-specific data
    
    -- Foreign key constraint
    FOREIGN KEY(trip_id) REFERENCES trips(id) ON DELETE CASCADE
);

-- CHAT HISTORY TABLE
CREATE TABLE chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trip_id INTEGER, -- Allow NULL for pre-trip conversations
    user_id INTEGER NOT NULL, -- Track conversations even without trips
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    phase TEXT CHECK(phase IN ('phase1_langflow', 'phase2_crewai', 'phase3_autogen', 'phase4_langgraph')),
    content TEXT NOT NULL,
    metadata TEXT, -- JSON for additional data
    sequence_number INTEGER, -- Order within conversation
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign keys
    FOREIGN KEY(trip_id) REFERENCES trips(id),
    FOREIGN KEY(user_id) REFERENCES users(id)
);

-- PERFORMANCE INDEXES
CREATE INDEX idx_trips_user_status ON trips(user_id, trip_status, created_at);
CREATE INDEX idx_trips_phase ON trips(phase, created_at);
CREATE INDEX idx_trips_dates ON trips(trip_startdate, trip_enddate);
CREATE INDEX idx_trip_plans_trip_id ON trip_plans(trip_id, version DESC);
CREATE INDEX idx_trip_plans_status ON trip_plans(status, generated_at);
CREATE INDEX idx_chat_trip_sequence ON chat_history(trip_id, sequence_number);

-- TRIGGERS FOR AUTO-UPDATE
CREATE TRIGGER update_trips_timestamp 
    AFTER UPDATE ON trips
    BEGIN
        UPDATE trips SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;

CREATE TRIGGER update_trip_plans_timestamp 
    AFTER UPDATE ON trip_plans
    BEGIN
        UPDATE trip_plans SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;


-- VIEWS FOR COMMON QUERIES
CREATE VIEW active_trips AS
SELECT 
    t.*,
    u.name as user_name,
    u.email as user_email,
    (julianday(t.trip_enddate) - julianday(t.trip_startdate) + 1) as duration_days
FROM trips t
JOIN users u ON t.user_id = u.id
WHERE t.trip_status IN ('draft', 'confirmed', 'in_progress');

CREATE VIEW trip_summary AS
SELECT 
    t.id,
    t.title,
    t.phase,
    u.name as user_name,
    t.origin || ' ? ' || t.destination as route,
    t.trip_startdate,
    t.trip_enddate,
    t.no_of_adults + t.no_of_children as total_travelers,
    t.budget || ' ' || t.currency as budget_display,
    t.trip_status,
    COUNT(ch.id) as message_count,
    tp.status as plan_status,
    tp.total_estimated_cost
FROM trips t
JOIN users u ON t.user_id = u.id
LEFT JOIN chat_history ch ON t.id = ch.trip_id
LEFT JOIN trip_plans tp ON t.id = tp.trip_id AND tp.version = (
    SELECT MAX(version) FROM trip_plans WHERE trip_id = t.id
)
GROUP BY t.id;

CREATE VIEW trip_plans_with_details AS
SELECT 
    tp.*,
    t.title as trip_title,
    t.origin,
    t.destination,
    t.trip_startdate,
    t.trip_enddate,
    u.name as user_name
FROM trip_plans tp
JOIN trips t ON tp.trip_id = t.id
JOIN users u ON t.user_id = u.id;

-- ADMIN QUERIES FOR MONITORING
-- SELECT * FROM trip_summary WHERE trip_status = 'draft';
-- SELECT phase, COUNT(*) FROM trips GROUP BY phase;