import os
import csv
import json
import io
from flask import (
    Flask, render_template, request, redirect, url_for, 
    flash, jsonify, Response, send_file, abort
)
from werkzeug.utils import secure_filename

from config import (
    UPLOAD_FOLDER, EVIDENCE_FOLDER, ALLOWED_EXTENSIONS, 
    MAX_CONTENT_LENGTH, SECRET_KEY
)
import database
import detector

# Initialize Flask Application
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure database tables exist upon startup
database.init_db()

def allowed_file(filename):
    """Check if uploaded file has an authorized video extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ----------------- Dashboard & Web Routes -----------------

@app.route('/')
def dashboard():
    """Main dashboard displaying summary analytics, model statuses, and recent records."""
    stats = database.get_dashboard_stats()
    model_status = detector.get_system_model_status()
    all_videos = database.get_all_videos()
    return render_template(
        'index.html',
        stats=stats,
        model_status=model_status,
        videos=all_videos[:5]
    )

@app.route('/upload', methods=['GET', 'POST'])
def upload_video():
    """Handles traffic video uploads with validation and directs to processing."""
    if request.method == 'POST':
        if 'video_file' not in request.files:
            flash('No file part provided in request.', 'error')
            return redirect(request.url)
        
        file = request.files['video_file']
        if file.filename == '':
            flash('No video selected for upload.', 'error')
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash(f"Invalid file type. Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS)).upper()}", 'error')
            return redirect(request.url)

        # Secure filename and save
        original_name = file.filename
        safe_name = secure_filename(file.filename)
        timestamp_prefix = int(database.datetime.now().timestamp())
        saved_filename = f"{timestamp_prefix}_{safe_name}"
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        
        try:
            file.save(saved_path)
            # Create database video record
            video_id = database.add_video(saved_filename, original_name, saved_path)
            flash(f"Video '{original_name}' uploaded successfully. Ready for AI inspection.", 'success')
            return redirect(url_for('process_view', video_id=video_id))
        except Exception as e:
            flash(f"Failed to save uploaded video: {str(e)}", 'error')
            return redirect(request.url)

    model_status = detector.get_system_model_status()
    return render_template('upload.html', model_status=model_status)

@app.route('/process/<int:video_id>')
def process_view(video_id):
    """View rendering real-time video processing progress and live telemetry."""
    video = database.get_video(video_id)
    if not video:
        flash("Video not found.", "error")
        return redirect(url_for('dashboard'))
    
    model_status = detector.get_system_model_status()
    return render_template('process.html', video=video, model_status=model_status)

@app.route('/stream_process/<int:video_id>')
def stream_process(video_id):
    """Server-Sent Events (SSE) route streaming detection progress in real-time."""
    def generate():
        for update in detector.process_video_generator(video_id):
            yield f"data: {json.dumps(update)}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )

@app.route('/records')
def records_view():
    """Filterable and searchable detection records and violation evidence gallery."""
    search = request.args.get('search', '').strip()
    event_filter = request.args.get('event', 'all')
    status_filter = request.args.get('status', 'all')
    vehicle_filter = request.args.get('vehicle', 'all')
    video_id = request.args.get('video_id', None, type=int)

    page = request.args.get('page', 1, type=int)
    limit = 25
    offset = (page - 1) * limit

    records = database.get_violations(
        search=search if search else None,
        event_filter=event_filter,
        status_filter=status_filter,
        vehicle_filter=vehicle_filter,
        video_id=video_id,
        limit=limit,
        offset=offset
    )

    total_records = database.get_violations_count(
        search=search if search else None,
        event_filter=event_filter,
        status_filter=status_filter,
        vehicle_filter=vehicle_filter,
        video_id=video_id
    )

    total_pages = (total_records + limit - 1) // limit if total_records > 0 else 1
    all_videos = database.get_all_videos()

    return render_template(
        'results.html',
        records=records,
        search=search,
        event_filter=event_filter,
        status_filter=status_filter,
        vehicle_filter=vehicle_filter,
        selected_video_id=video_id,
        all_videos=all_videos,
        page=page,
        total_pages=total_pages,
        total_records=total_records
    )

@app.route('/videos')
def videos_view():
    """Listing of all uploaded traffic videos and processing states."""
    videos = database.get_all_videos()
    return render_template('videos.html', videos=videos)

# ----------------- Review & API Endpoints -----------------

@app.route('/api/update_review', methods=['POST'])
def api_update_review():
    """AJAX endpoint for traffic officers/reviewers to approve, dismiss, or hold records."""
    data = request.get_json() or {}
    violation_id = data.get('violation_id')
    new_status = data.get('review_status')
    notes = data.get('notes')

    valid_statuses = {'Pending Review', 'Approved Violation', 'Dismissed / False Positive'}
    if not violation_id or new_status not in valid_statuses:
        return jsonify({"success": False, "error": "Invalid violation ID or review status"}), 400

    updated = database.update_violation_review(violation_id, new_status, notes)
    if updated:
        return jsonify({"success": True, "violation_id": violation_id, "new_status": new_status})
    return jsonify({"success": False, "error": "Record not found"}), 404

@app.route('/api/stats')
def api_stats():
    """JSON API providing aggregated metrics for Chart.js visual analytics."""
    stats = database.get_dashboard_stats()
    return jsonify(stats)

@app.route('/api/model_status')
def api_model_status():
    """JSON API detailing model availability and configurations."""
    status = detector.get_system_model_status()
    return jsonify(status)

@app.route('/export_csv')
def export_csv():
    """Exports all violation records to a downloadable CSV spreadsheet."""
    records = database.export_violations_data()
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        'Record ID', 'Video Name', 'Event Type', 'Vehicle Type', 
        'Timestamp (Time)', 'Number Plate', 'Detection Confidence', 
        'Human Review Status', 'Reviewer Notes', 'Logged At'
    ])

    for r in records:
        writer.writerow([
            r['id'],
            r['video_name'],
            r['event_type'],
            r['vehicle_type'],
            r['formatted_time'],
            r['plate_text'],
            f"{r['confidence'] * 100:.1f}%",
            r['review_status'],
            r['reviewer_notes'],
            r['created_at']
        ])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=traffic_violations_export.csv"}
    )

# ----------------- Error Handlers -----------------

@app.errorhandler(413)
def request_entity_too_large(error):
    max_mb = MAX_CONTENT_LENGTH // (1024 * 1024)
    flash(f"The uploaded video is too large. Maximum supported file size is {max_mb} MB.", "error")
    return redirect(url_for('upload_video'))

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Local development server on port 5000
    print("[INFO] Starting Traffic Violation System server at http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True)
