import os
import time
import math
import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    VEHICLE_MODEL_PATH,
    HELMET_MODEL_PATH,
    PLATE_MODEL_PATH,
    VEHICLE_CONF_THRESHOLD,
    HELMET_CONF_THRESHOLD,
    FRAME_SKIP,
    DUPLICATE_DEBOUNCE_SECONDS,
    COCO_VEHICLE_CLASSES,
    EVIDENCE_FOLDER
)
import database
import plate_reader

# Global model caches
_vehicle_yolo_model = None
_helmet_yolo_model = None

def get_vehicle_model():
    """Load or return cached genuine YOLOv8 vehicle detection model."""
    global _vehicle_yolo_model
    if _vehicle_yolo_model is None:
        if not os.path.exists(VEHICLE_MODEL_PATH):
            raise FileNotFoundError(f"Vehicle model file not found at {VEHICLE_MODEL_PATH}")
        print(f"[INFO] Loading vehicle detection model from {VEHICLE_MODEL_PATH}...")
        _vehicle_yolo_model = YOLO(VEHICLE_MODEL_PATH)
    return _vehicle_yolo_model

def is_helmet_model_available():
    """Check if custom helmet detection model file exists on disk."""
    return os.path.exists(HELMET_MODEL_PATH)

def get_helmet_model():
    """Load custom helmet detection model if weights are available."""
    global _helmet_yolo_model
    if not is_helmet_model_available():
        return None
    if _helmet_yolo_model is None:
        try:
            print(f"[INFO] Loading helmet detection model from {HELMET_MODEL_PATH}...")
            _helmet_yolo_model = YOLO(HELMET_MODEL_PATH)
        except Exception as e:
            print(f"[WARNING] Could not load helmet model: {e}")
            _helmet_yolo_model = None
    return _helmet_yolo_model

def get_system_model_status():
    """
    Returns an honest, complete overview of all computer vision models
    and their operational status.
    """
    veh_available = os.path.exists(VEHICLE_MODEL_PATH)
    helmet_available = is_helmet_model_available()
    plate_status = plate_reader.get_plate_model_status()

    return {
        "vehicle_model": {
            "name": "YOLOv8 Nano (Pretrained COCO)",
            "file": os.path.basename(VEHICLE_MODEL_PATH),
            "available": veh_available,
            "status": "Operational" if veh_available else "Missing Weights",
            "supported_classes": ["car", "motorcycle", "bus", "truck"]
        },
        "helmet_model": {
            "name": "YOLO Helmet Detector (Ultralytics)",
            "file": os.path.basename(HELMET_MODEL_PATH),
            "available": helmet_available,
            "status": "Operational" if helmet_available else "Not Installed",
            "message": (
                "YOLO helmet detection model active (detects helmet vs. no-helmet)." 
                if helmet_available else 
                "Place 'helmet_model.pt' into models/ to enable direct helmet vs. no-helmet classification."
            )
        },
        "plate_model": {
            "name": "YOLO License Plate Detector",
            "file": os.path.basename(PLATE_MODEL_PATH),
            "available": plate_status["model_file_exists"],
            "status": "Operational" if plate_status["model_file_exists"] else "Morphology Fallback",
            "ocr_engine": plate_status["ocr_engine"],
            "ocr_available": plate_status["ocr_available"],
            "message": plate_status["status_message"]
        }
    }

def format_timestamp(seconds):
    """Format seconds into readable MM:SS.s string."""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:04.1f}"

def calculate_iou(boxA, boxB):
    """Calculate Intersection over Union (IoU) of two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

class DebounceTracker:
    """
    Prevents duplicate evidence capture for the same vehicle in consecutive frames.
    Tracks bounding boxes and timestamps.
    """
    def __init__(self, debounce_seconds=DUPLICATE_DEBOUNCE_SECONDS, iou_threshold=0.45):
        self.debounce_seconds = debounce_seconds
        self.iou_threshold = iou_threshold
        self.tracked_vehicles = [] # list of dicts: {'box': [x1, y1, x2, y2], 'class': str, 'timestamp': float}

    def is_duplicate(self, new_box, vehicle_type, current_timestamp):
        # Purge stale records older than debounce window
        self.tracked_vehicles = [
            t for t in self.tracked_vehicles 
            if (current_timestamp - t['timestamp']) < self.debounce_seconds
        ]

        for item in self.tracked_vehicles:
            if item['class'] == vehicle_type:
                iou = calculate_iou(new_box, item['box'])
                if iou > self.iou_threshold:
                    # Refresh timestamp to keep tracking
                    item['timestamp'] = current_timestamp
                    item['box'] = new_box
                    return True

        # Register new tracked instance
        self.tracked_vehicles.append({
            'box': new_box,
            'class': vehicle_type,
            'timestamp': current_timestamp
        })
        return False

def draw_hud_and_evidence(frame, vehicle_box, vehicle_type, event_type, 
                          confidence, timestamp_str, plate_text="N/A", 
                          plate_box=None, rider_box=None, is_violation=False,
                          helmet_boxes=None, crop_offset=(0, 0)):
    """
    Renders clean, professional annotations on the evidence frame:
    - Vehicle bounding box with label
    - License plate highlight box and text
    - Helmet / No-Helmet detection bounding box on the rider's head
    - Top HUD telemetry banner with timestamp, event, and confidence
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Color palette (BGR)
    # Red for violations, Gold for pending review, Emerald for normal
    box_color = (48, 59, 235) if is_violation else (40, 180, 80)
    plate_color = (255, 191, 0) # Deep sky blue / amber

    # 1. Draw vehicle bounding box
    vx1, vy1, vx2, vy2 = vehicle_box
    cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), box_color, 2)

    # Vehicle label pill
    label = f"{vehicle_type.upper()} ({confidence * 100:.0f}%)"
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(annotated, (vx1, max(0, vy1 - th - 8)), (vx1 + tw + 10, vy1), box_color, -1)
    cv2.putText(annotated, label, (vx1 + 5, max(0, vy1 - 5)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # 2. Draw plate bounding box if detected
    if plate_box is not None:
        px1, py1, px2, py2 = plate_box
        # Offset to vehicle coordinate
        abs_px1 = vx1 + px1
        abs_py1 = vy1 + py1
        abs_px2 = vx1 + px2
        abs_py2 = vy1 + py2
        cv2.rectangle(annotated, (abs_px1, abs_py1), (abs_px2, abs_py2), plate_color, 2)
        plate_badge = f"PLATE: {plate_text}"
        cv2.putText(annotated, plate_badge, (abs_px1, max(0, abs_py1 - 5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, plate_color, 2, cv2.LINE_AA)

    # 3. Draw helmet / no-helmet bounding boxes on rider
    if helmet_boxes:
        ox, oy = crop_offset
        for hx1, hy1, hx2, hy2, h_label, h_conf, is_compliant in helmet_boxes:
            abs_hx1 = ox + hx1
            abs_hy1 = oy + hy1
            abs_hx2 = ox + hx2
            abs_hy2 = oy + hy2
            h_color = (40, 180, 80) if is_compliant else (48, 59, 235) # Emerald if compliant, Crimson if violation
            cv2.rectangle(annotated, (abs_hx1, abs_hy1), (abs_hx2, abs_hy2), h_color, 2)
            h_badge = f"{h_label} ({h_conf * 100:.0f}%)"
            (htw, hth), _ = cv2.getTextSize(h_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (abs_hx1, max(0, abs_hy1 - hth - 6)), (abs_hx1 + htw + 6, abs_hy1), h_color, -1)
            cv2.putText(annotated, h_badge, (abs_hx1 + 3, max(0, abs_hy1 - 4)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # 4. Draw Top HUD Telemetry Bar
    hud_height = 42
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_height), (15, 23, 42), -1) # Dark navy HUD
    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

    # Top border line
    banner_color = (0, 70, 220) if is_violation else (0, 180, 80)
    cv2.line(annotated, (0, hud_height), (w, hud_height), banner_color, 2)

    # HUD Text items
    status_tag = "[VIOLATION SUSPECTED]" if is_violation else "[MONITORED VEHICLE]"
    hud_text_left = f"{status_tag} {event_type.upper()}"
    hud_text_right = f"TIME: {timestamp_str} | CONF: {confidence * 100:.1f}% | PLATE: {plate_text}"

    cv2.putText(annotated, hud_text_left, (15, 26), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    
    (rw, _), _ = cv2.getTextSize(hud_text_right, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(annotated, hud_text_right, (max(15, w - rw - 15), 26), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 240), 1, cv2.LINE_AA)

    return annotated

def process_video_generator(video_id):
    """
    Generator function that runs detection on an uploaded video frame-by-frame.
    Yields progress dicts for live UI streaming and saves evidence & DB records.
    """
    video_record = database.get_video(video_id)
    if not video_record:
        yield {"error": "Video record not found in database", "done": True}
        return

    video_path = video_record['file_path']
    if not os.path.exists(video_path):
        database.update_video_status(video_id, 'failed', error_message="Video file missing on disk")
        yield {"error": f"Video file not found at {video_path}", "done": True}
        return

    # Update DB status to processing
    database.update_video_status(video_id, 'processing')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        database.update_video_status(video_id, 'failed', error_message="Could not open video with OpenCV")
        yield {"error": "Failed to open video file. File might be corrupted.", "done": True}
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration_sec = total_frames / video_fps if total_frames > 0 else 0.0

    vehicle_model = get_vehicle_model()
    helmet_model = get_helmet_model()
    helmet_available = (helmet_model is not None)

    tracker = DebounceTracker(debounce_seconds=DUPLICATE_DEBOUNCE_SECONDS)
    frame_idx = 0
    violations_detected = 0
    start_time = time.time()

    yield {
        "status": "processing",
        "current_frame": 0,
        "total_frames": total_frames,
        "percent": 0,
        "violations": 0,
        "fps": 0.0,
        "done": False,
        "message": "Starting video analysis..."
    }

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Frame skip optimization for smooth processing
            if frame_idx % FRAME_SKIP != 0:
                continue

            current_timestamp_sec = frame_idx / video_fps
            formatted_time = format_timestamp(current_timestamp_sec)

            # Run YOLO vehicle detection
            results = vehicle_model(frame, conf=VEHICLE_CONF_THRESHOLD, verbose=False)

            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    if cls_id not in COCO_VEHICLE_CLASSES:
                        continue

                    vehicle_type = COCO_VEHICLE_CLASSES[cls_id]
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    h_frame, w_frame = frame.shape[:2]
                    
                    # Ensure bbox inside bounds
                    x1 = max(0, min(x1, w_frame - 1))
                    y1 = max(0, min(y1, h_frame - 1))
                    x2 = max(0, min(x2, w_frame))
                    y2 = max(0, min(y2, h_frame))

                    if (x2 - x1) < 20 or (y2 - y1) < 20:
                        continue

                    # Check debounce to prevent saving identical duplicates
                    if tracker.is_duplicate([x1, y1, x2, y2], vehicle_type, current_timestamp_sec):
                        continue

                    vehicle_crop = frame[y1:y2, x1:x2]

                    # 1. Number plate extraction via OCR
                    plate_res = plate_reader.detect_and_read_plate(vehicle_crop)
                    plate_text = plate_res.get('plate_text', 'N/A')
                    plate_box = plate_res.get('box', None)

                    # 2. Determine Event Type & Violation Status
                    event_type = f"Vehicle Detected - {vehicle_type.capitalize()}"
                    is_violation = False
                    review_status = "Pending Review"
                    notes = ""
                    detected_helmet_boxes = []
                    crop_offset = (x1, y1)

                    # Special handling for motorcycles (helmet compliance check)
                    if vehicle_type == 'motorcycle':
                        if helmet_available:
                            # Run custom helmet model on motorcycle crop
                            try:
                                # Expand motorcycle crop upwards by 35% to include rider torso & head
                                crop_top = max(0, y1 - int((y2 - y1) * 0.35))
                                rider_crop = frame[crop_top:y2, x1:x2]
                                crop_offset = (x1, crop_top)

                                helmet_res = helmet_model(rider_crop, conf=HELMET_CONF_THRESHOLD, verbose=False)
                                has_helmet = False
                                has_no_helmet = False

                                for hr in helmet_res:
                                    for h_box in hr.boxes:
                                        h_conf = float(h_box.conf[0])
                                        h_cls_id = int(h_box.cls[0])
                                        h_cls_name = helmet_model.names.get(h_cls_id, "").lower()
                                        hx1, hy1, hx2, hy2 = [int(v) for v in h_box.xyxy[0].tolist()]

                                        # Flexible matching for common helmet dataset classes
                                        # (Hardhat / NO-Hardhat, helmet / no_helmet, with_helmet / without_helmet)
                                        if any(neg in h_cls_name for neg in ["no", "without", "bare"]):
                                            has_no_helmet = True
                                            detected_helmet_boxes.append((hx1, hy1, hx2, hy2, "NO HELMET", h_conf, False))
                                        elif any(pos in h_cls_name for pos in ["helmet", "hardhat", "with", "wear"]):
                                            has_helmet = True
                                            detected_helmet_boxes.append((hx1, hy1, hx2, hy2, "HELMET", h_conf, True))

                                if has_no_helmet:
                                    event_type = "No Helmet Detected"
                                    is_violation = True
                                    review_status = "Pending Review"
                                    notes = "AI Helmet Detection: Motorcycle rider detected without helmet. Flagged for review."
                                elif has_helmet:
                                    event_type = "Helmet Detected"
                                    is_violation = False
                                    review_status = "Pending Review"
                                    notes = "AI Helmet Detection: Motorcycle rider wearing compliant helmet."
                                else:
                                    event_type = "Motorcycle Monitored (Helmet Inconclusive)"
                                    is_violation = False
                                    review_status = "Pending Review"
                                    notes = "Helmet status inconclusive from viewing angle. Flagged for manual audit."
                            except Exception as he:
                                print(f"[WARNING] Helmet inference error: {he}")
                                event_type = "Motorcycle Monitored"
                        else:
                            # Handling when helmet weights are not yet installed
                            event_type = "Motorcycle Monitored (No Helmet Model Installed)"
                            is_violation = False
                            notes = "Helmet model not installed. Flagged for manual review."

                    # 3. Save timestamped evidence image
                    evidence_filename = f"evidence_{video_id}_{frame_idx}_{int(time.time() * 1000) % 100000}.jpg"
                    evidence_path = os.path.join(EVIDENCE_FOLDER, evidence_filename)

                    evidence_frame = draw_hud_and_evidence(
                        frame=frame,
                        vehicle_box=[x1, y1, x2, y2],
                        vehicle_type=vehicle_type,
                        event_type=event_type,
                        confidence=conf,
                        timestamp_str=formatted_time,
                        plate_text=plate_text,
                        plate_box=plate_box,
                        is_violation=is_violation,
                        helmet_boxes=detected_helmet_boxes,
                        crop_offset=crop_offset
                    )

                    cv2.imwrite(evidence_path, evidence_frame)

                    # 4. Insert into SQLite database
                    database.add_violation(
                        video_id=video_id,
                        video_name=video_record['original_name'],
                        event_type=event_type,
                        vehicle_type=vehicle_type,
                        timestamp_sec=current_timestamp_sec,
                        formatted_time=formatted_time,
                        plate_text=plate_text,
                        confidence=round(conf, 3),
                        evidence_image=evidence_filename,
                        review_status=review_status,
                        reviewer_notes=notes
                    )

                    violations_detected += 1

            # Yield live progress update periodically
            elapsed = time.time() - start_time
            current_fps = round(frame_idx / max(elapsed, 0.001), 1)
            pct = int((frame_idx / max(total_frames, 1)) * 100)

            if frame_idx % (FRAME_SKIP * 2) == 0 or frame_idx >= total_frames:
                yield {
                    "status": "processing",
                    "current_frame": frame_idx,
                    "total_frames": total_frames,
                    "percent": min(pct, 99),
                    "violations": violations_detected,
                    "fps": current_fps,
                    "done": False,
                    "message": f"Processed frame {frame_idx}/{total_frames} ({pct}%)"
                }

    except Exception as e:
        print(f"[ERROR] Exception during video processing: {e}")
        database.update_video_status(video_id, 'failed', error_message=str(e))
        yield {"error": f"Processing interrupted: {str(e)}", "done": True}
        return
    finally:
        cap.release()

    # Mark video completed in DB
    database.update_video_status(
        video_id=video_id,
        status='completed',
        total_frames=total_frames,
        processed_frames=frame_idx,
        duration=duration_sec
    )

    yield {
        "status": "completed",
        "current_frame": total_frames,
        "total_frames": total_frames,
        "percent": 100,
        "violations": violations_detected,
        "fps": round(frame_idx / max(time.time() - start_time, 0.001), 1),
        "done": True,
        "message": f"Video processing completed. Captured {violations_detected} detection events."
    }
