import os
import re
import cv2
import numpy as np
from config import PLATE_MODEL_PATH, PLATE_CONF_THRESHOLD

# Global holders for lazy-loaded models
_plate_yolo_model = None
_easyocr_reader = None
_ocr_initialized = False
_ocr_available = False

def is_plate_model_available():
    """Check if the dedicated YOLO plate detection model file exists on disk."""
    return os.path.exists(PLATE_MODEL_PATH)

def get_plate_yolo_model():
    """Lazy load the dedicated YOLO license plate model if the weights file is present."""
    global _plate_yolo_model
    if not is_plate_model_available():
        return None
    if _plate_yolo_model is None:
        try:
            from ultralytics import YOLO
            _plate_yolo_model = YOLO(PLATE_MODEL_PATH)
            print(f"[INFO] Successfully loaded dedicated License Plate YOLO model from {PLATE_MODEL_PATH}")
        except Exception as e:
            print(f"[WARNING] Failed to load License Plate model from {PLATE_MODEL_PATH}: {e}")
            _plate_yolo_model = None
    return _plate_yolo_model

def get_easyocr_reader():
    """
    Lazy load the EasyOCR reader. 
    Handles offline conditions or initial download gracefully.
    """
    global _easyocr_reader, _ocr_initialized, _ocr_available
    if _ocr_initialized:
        return _easyocr_reader

    _ocr_initialized = True
    try:
        import easyocr
        import torch
        use_gpu = torch.cuda.is_available()
        print(f"[INFO] Initializing EasyOCR (GPU acceleration: {use_gpu})...")
        # Initialize English OCR reader
        _easyocr_reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
        _ocr_available = True
        print("[INFO] EasyOCR initialized successfully.")
    except Exception as e:
        print(f"[WARNING] EasyOCR could not be initialized: {e}")
        _easyocr_reader = None
        _ocr_available = False

    return _easyocr_reader

def clean_plate_text(raw_text):
    """
    Clean extracted OCR text to match standard license plate alphanumeric formatting.
    Removes symbols, extra spaces, and common OCR noise.
    """
    if not raw_text:
        return "N/A"
    
    # Strip symbols and leave only alphanumeric characters
    cleaned = re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()
    
    # Standard license plates are typically between 4 and 12 characters
    if len(cleaned) < 3:
        return "N/A"
    
    return cleaned

def preprocess_plate_image(plate_crop):
    """
    Preprocess license plate crop for improved OCR character extraction:
    - Grayscale conversion
    - Contrast enhancement (CLAHE)
    - Bilateral filter for noise reduction while keeping edges sharp
    - Adaptive thresholding
    """
    if plate_crop is None or plate_crop.size == 0:
        return None

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    
    # Resize if too small to assist OCR
    h, w = gray.shape
    if h < 40 or w < 100:
        scale = max(40 / max(h, 1), 100 / max(w, 1), 2.0)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # Contrast Limiting Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Denoise while preserving edges
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    return denoised

def locate_plate_morphological_roi(vehicle_crop):
    """
    Morphology-based heuristic fallback to locate horizontal rectangular license plate 
    candidates when a dedicated trained YOLO plate model is not yet provided.
    """
    if vehicle_crop is None or vehicle_crop.size == 0:
        return None, None

    vh, vw = vehicle_crop.shape[:2]
    # Restrict search to bottom 60% of the vehicle where plates are positioned
    roi_y1 = int(vh * 0.4)
    search_region = vehicle_crop[roi_y1:, :]
    if search_region.size == 0:
        return None, None

    gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
    # Sobel edge filter
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    grad_x = cv2.convertScaleAbs(grad_x)

    # Morphological close with horizontal rectangular kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    closed = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, kernel)
    
    # Thresholding
    _, thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_candidate = None
    best_aspect_score = 0
    best_bbox = None

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue
        aspect_ratio = float(w) / h
        # Typical license plates have an aspect ratio of 2.0 to 5.5
        area = w * h
        if 2.0 <= aspect_ratio <= 5.8 and 500 <= area <= (vw * vh * 0.15):
            score = area * (1.0 / (abs(aspect_ratio - 3.5) + 1.0))
            if score > best_aspect_score:
                best_aspect_score = score
                best_candidate = search_region[y:y+h, x:x+w]
                best_bbox = (x, roi_y1 + y, x + w, roi_y1 + y + h)

    return best_candidate, best_bbox

def extract_plate_text_from_crop(plate_crop):
    """
    Runs EasyOCR on a cropped plate image and returns (cleaned_text, confidence).
    """
    reader = get_easyocr_reader()
    if reader is None or plate_crop is None or plate_crop.size == 0:
        return "N/A", 0.0

    try:
        # Preprocessing
        processed = preprocess_plate_image(plate_crop)
        if processed is None:
            return "N/A", 0.0

        # Perform OCR recognition
        results = reader.readtext(processed, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        if not results:
            # Retry on raw crop if processed yielded no results
            results = reader.readtext(plate_crop, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

        if not results:
            return "N/A", 0.0

        # Aggregate text and confidence
        detected_texts = []
        confidences = []
        for bbox, text, conf in results:
            cleaned = clean_plate_text(text)
            if cleaned != "N/A":
                detected_texts.append(cleaned)
                confidences.append(conf)

        if detected_texts:
            full_text = "".join(detected_texts)
            avg_conf = sum(confidences) / len(confidences)
            return full_text, round(float(avg_conf), 2)
            
    except Exception as e:
        print(f"[ERROR] OCR processing error: {e}")

    return "N/A", 0.0

def detect_and_read_plate(vehicle_crop):
    """
    Main license plate pipeline:
    1. Check for dedicated YOLO plate model.
    2. Fallback to morphological plate ROI locator if model is missing.
    3. Run EasyOCR on candidate plate ROI.
    Returns:
        dict: {
            "plate_text": str,
            "confidence": float,
            "box": tuple or None,
            "method": str,
            "is_model_backed": bool
        }
    """
    if vehicle_crop is None or vehicle_crop.size == 0:
        return {
            "plate_text": "N/A",
            "confidence": 0.0,
            "box": None,
            "method": "None",
            "is_model_backed": False
        }

    yolo_model = get_plate_yolo_model()

    # Path A: Dedicated Trained YOLO License Plate Model Available
    if yolo_model is not None:
        try:
            results = yolo_model(vehicle_crop, conf=PLATE_CONF_THRESHOLD, verbose=False)
            for r in results:
                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    best_box = boxes[0]
                    x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
                    plate_crop = vehicle_crop[y1:y2, x1:x2]
                    text, ocr_conf = extract_plate_text_from_crop(plate_crop)
                    return {
                        "plate_text": text,
                        "confidence": ocr_conf if text != "N/A" else float(best_box.conf[0]),
                        "box": (x1, y1, x2, y2),
                        "method": "Dedicated YOLO Plate Model + EasyOCR",
                        "is_model_backed": True
                    }
        except Exception as e:
            print(f"[ERROR] Dedicated YOLO plate inference error: {e}")

    # Path B: Morphological Candidate ROI + EasyOCR Fallback
    candidate_crop, bbox = locate_plate_morphological_roi(vehicle_crop)
    if candidate_crop is not None and candidate_crop.size > 0:
        text, ocr_conf = extract_plate_text_from_crop(candidate_crop)
        if text != "N/A":
            return {
                "plate_text": text,
                "confidence": ocr_conf,
                "box": bbox,
                "method": "Morphological ROI + EasyOCR",
                "is_model_backed": False
            }

    return {
        "plate_text": "N/A",
        "confidence": 0.0,
        "box": None,
        "method": "None",
        "is_model_backed": False
    }

def get_plate_model_status():
    """Return model and OCR availability information for the dashboard and API."""
    model_exists = is_plate_model_available()
    reader = get_easyocr_reader()
    ocr_ready = (reader is not None)

    return {
        "model_file_exists": model_exists,
        "model_path": PLATE_MODEL_PATH,
        "ocr_engine": "EasyOCR (PyTorch)",
        "ocr_available": ocr_ready,
        "mode": "Dedicated YOLO Plate Model" if model_exists else "Morphological ROI + EasyOCR Baseline",
        "status_message": (
            "Dedicated license plate YOLO model active." 
            if model_exists else 
            "Dedicated model 'plate_model.pt' not found. Operating with Morphological ROI locator + EasyOCR fallback."
        )
    }
