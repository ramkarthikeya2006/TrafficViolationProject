import sqlite3
import os
from datetime import datetime
from config import DATABASE_PATH

def get_db_connection():
    """Establish and return a connection to the SQLite database with row factory."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    """Initialize database tables with schema migrations and indexes."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Table for uploaded traffic videos
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS videos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        original_name TEXT NOT NULL,
        file_path TEXT NOT NULL,
        upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        duration REAL DEFAULT 0,
        total_frames INTEGER DEFAULT 0,
        processed_frames INTEGER DEFAULT 0,
        status TEXT DEFAULT 'pending', -- pending, processing, completed, failed
        error_message TEXT
    )
    """)

    # Table for detection events and suspected traffic violations
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS violations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id INTEGER,
        video_name TEXT NOT NULL,
        event_type TEXT NOT NULL,       -- e.g., 'No Helmet Detected', 'Helmet Detected', 'Vehicle Detected'
        vehicle_type TEXT NOT NULL,     -- e.g., 'motorcycle', 'car', 'bus', 'truck'
        timestamp_sec REAL NOT NULL,    -- video timestamp in seconds
        formatted_time TEXT NOT NULL,   -- e.g., '00:14.2'
        plate_text TEXT DEFAULT 'N/A',  -- Extracted number plate string
        confidence REAL NOT NULL,       -- Model detection confidence (0.0 to 1.0)
        evidence_image TEXT NOT NULL,   -- Saved evidence image filename
        review_status TEXT DEFAULT 'Pending Review', -- 'Pending Review', 'Approved Violation', 'Dismissed / False Positive'
        reviewer_notes TEXT DEFAULT '',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (video_id) REFERENCES videos (id) ON DELETE SET NULL
    )
    """)

    # Create indexes for fast filtering and searching
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_violations_event ON violations(event_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_violations_review ON violations(review_status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_violations_vehicle ON violations(vehicle_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_violations_video ON violations(video_id)")

    conn.commit()
    conn.close()

# ----------------- Video Operations -----------------

def add_video(filename, original_name, file_path):
    """Insert a new video record and return its primary key ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO videos (filename, original_name, file_path, status) VALUES (?, ?, ?, 'pending')",
        (filename, original_name, file_path)
    )
    video_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return video_id

def update_video_status(video_id, status, error_message=None, total_frames=0, processed_frames=0, duration=0.0):
    """Update video processing status, metrics, and any error message."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE videos 
        SET status = ?, error_message = ?, total_frames = ?, processed_frames = ?, duration = ?
        WHERE id = ?
        """,
        (status, error_message, total_frames, processed_frames, duration, video_id)
    )
    conn.commit()
    conn.close()

def get_video(video_id):
    """Fetch single video by ID."""
    conn = get_db_connection()
    video = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
    conn.close()
    return dict(video) if video else None

def get_all_videos():
    """Retrieve all uploaded videos ordered by most recent upload."""
    conn = get_db_connection()
    videos = conn.execute("SELECT * FROM videos ORDER BY upload_time DESC").fetchall()
    conn.close()
    return [dict(v) for v in videos]

# ----------------- Violation / Detection Operations -----------------

def add_violation(video_id, video_name, event_type, vehicle_type, timestamp_sec, 
                  formatted_time, plate_text, confidence, evidence_image, 
                  review_status='Pending Review', reviewer_notes=''):
    """Insert a new detection or violation record."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO violations 
        (video_id, video_name, event_type, vehicle_type, timestamp_sec, formatted_time, 
         plate_text, confidence, evidence_image, review_status, reviewer_notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (video_id, video_name, event_type, vehicle_type, timestamp_sec, formatted_time,
         plate_text, confidence, evidence_image, review_status, reviewer_notes)
    )
    violation_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return violation_id

def get_violation(violation_id):
    """Fetch a single violation record by ID."""
    conn = get_db_connection()
    record = conn.execute("SELECT * FROM violations WHERE id = ?", (violation_id,)).fetchone()
    conn.close()
    return dict(record) if record else None

def get_violations(search=None, event_filter=None, status_filter=None, vehicle_filter=None, 
                   video_id=None, limit=100, offset=0):
    """
    Search and filter violations with safe parameterized queries.
    Supports filtering by event type, review status, vehicle type, or text search in plate / video name.
    """
    conn = get_db_connection()
    query = "SELECT * FROM violations WHERE 1=1"
    params = []

    if video_id is not None:
        query += " AND video_id = ?"
        params.append(video_id)

    if event_filter and event_filter != 'all':
        query += " AND event_type = ?"
        params.append(event_filter)

    if status_filter and status_filter != 'all':
        query += " AND review_status = ?"
        params.append(status_filter)

    if vehicle_filter and vehicle_filter != 'all':
        query += " AND vehicle_type = ?"
        params.append(vehicle_filter)

    if search:
        search_term = f"%{search.strip()}%"
        query += " AND (plate_text LIKE ? OR video_name LIKE ? OR reviewer_notes LIKE ?)"
        params.extend([search_term, search_term, search_term])

    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    records = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in records]

def get_violations_count(search=None, event_filter=None, status_filter=None, vehicle_filter=None, video_id=None):
    """Return total count of violations matching the filters for pagination."""
    conn = get_db_connection()
    query = "SELECT COUNT(*) as cnt FROM violations WHERE 1=1"
    params = []

    if video_id is not None:
        query += " AND video_id = ?"
        params.append(video_id)

    if event_filter and event_filter != 'all':
        query += " AND event_type = ?"
        params.append(event_filter)

    if status_filter and status_filter != 'all':
        query += " AND review_status = ?"
        params.append(status_filter)

    if vehicle_filter and vehicle_filter != 'all':
        query += " AND vehicle_type = ?"
        params.append(vehicle_filter)

    if search:
        search_term = f"%{search.strip()}%"
        query += " AND (plate_text LIKE ? OR video_name LIKE ? OR reviewer_notes LIKE ?)"
        params.extend([search_term, search_term, search_term])

    row = conn.execute(query, params).fetchone()
    conn.close()
    return row['cnt'] if row else 0

def update_violation_review(violation_id, review_status, reviewer_notes=None):
    """Update human review status and notes for a suspected violation."""
    conn = get_db_connection()
    cursor = conn.cursor()
    if reviewer_notes is not None:
        cursor.execute(
            "UPDATE violations SET review_status = ?, reviewer_notes = ? WHERE id = ?",
            (review_status, reviewer_notes, violation_id)
        )
    else:
        cursor.execute(
            "UPDATE violations SET review_status = ? WHERE id = ?",
            (review_status, violation_id)
        )
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0

def delete_violation(violation_id):
    """Delete a single violation record."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM violations WHERE id = ?", (violation_id,))
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0

# ----------------- Dashboard Analytics Operations -----------------

def get_dashboard_stats():
    """Calculate aggregated metrics and breakdown data for the dashboard."""
    conn = get_db_connection()
    
    # Total detection records
    total_detections = conn.execute("SELECT COUNT(*) as c FROM violations").fetchone()['c']
    
    # Suspected violations (violations that are pending review or marked as violation)
    suspected_violations = conn.execute(
        "SELECT COUNT(*) as c FROM violations WHERE event_type LIKE '%No Helmet%' OR event_type LIKE '%Violation%' OR review_status = 'Approved Violation'"
    ).fetchone()['c']

    # Total uploaded videos
    total_videos = conn.execute("SELECT COUNT(*) as c FROM videos").fetchone()['c']

    # Pending human reviews
    pending_reviews = conn.execute(
        "SELECT COUNT(*) as c FROM violations WHERE review_status = 'Pending Review'"
    ).fetchone()['c']

    # Approved violations
    approved_violations = conn.execute(
        "SELECT COUNT(*) as c FROM violations WHERE review_status = 'Approved Violation'"
    ).fetchone()['c']

    # Dismissed / False Positives
    dismissed_count = conn.execute(
        "SELECT COUNT(*) as c FROM violations WHERE review_status = 'Dismissed / False Positive'"
    ).fetchone()['c']

    # Vehicle types distribution
    vehicle_rows = conn.execute(
        "SELECT vehicle_type, COUNT(*) as cnt FROM violations GROUP BY vehicle_type"
    ).fetchall()
    vehicle_distribution = {row['vehicle_type']: row['cnt'] for row in vehicle_rows}

    # Event types distribution
    event_rows = conn.execute(
        "SELECT event_type, COUNT(*) as cnt FROM violations GROUP BY event_type"
    ).fetchall()
    event_distribution = {row['event_type']: row['cnt'] for row in event_rows}

    # Review status distribution
    status_rows = conn.execute(
        "SELECT review_status, COUNT(*) as cnt FROM violations GROUP BY review_status"
    ).fetchall()
    review_distribution = {row['review_status']: row['cnt'] for row in status_rows}

    # Recent 6 detections
    recent_records = conn.execute(
        "SELECT * FROM violations ORDER BY id DESC LIMIT 6"
    ).fetchall()

    conn.close()

    return {
        "total_detections": total_detections,
        "suspected_violations": suspected_violations,
        "total_videos": total_videos,
        "pending_reviews": pending_reviews,
        "approved_violations": approved_violations,
        "dismissed_count": dismissed_count,
        "vehicle_distribution": vehicle_distribution,
        "event_distribution": event_distribution,
        "review_distribution": review_distribution,
        "recent_records": [dict(r) for r in recent_records]
    }

def export_violations_data():
    """Retrieve all violations for CSV export."""
    conn = get_db_connection()
    records = conn.execute(
        "SELECT id, video_name, event_type, vehicle_type, formatted_time, plate_text, confidence, review_status, reviewer_notes, created_at FROM violations ORDER BY id ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in records]
