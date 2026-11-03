import os

# Base directory of the project
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Storage folders
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
EVIDENCE_FOLDER = os.path.join(BASE_DIR, 'static', 'evidence')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATABASE_PATH = os.path.join(BASE_DIR, 'traffic_system.db')

# Ensure necessary directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EVIDENCE_FOLDER, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Upload configuration
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_CONTENT_LENGTH = 150 * 1024 * 1024  # 150 MB max video file size

# Flask settings
SECRET_KEY = os.environ.get('SECRET_KEY', 'traffic-violation-capstone-secret-key-2026')

# Model file paths
VEHICLE_MODEL_PATH = os.path.join(MODELS_DIR, 'yolov8n.pt')
HELMET_MODEL_PATH = os.path.join(MODELS_DIR, 'helmet_model.pt')
PLATE_MODEL_PATH = os.path.join(MODELS_DIR, 'plate_model.pt')

# Computer Vision & Detection thresholds
VEHICLE_CONF_THRESHOLD = 0.40
HELMET_CONF_THRESHOLD = 0.40
PLATE_CONF_THRESHOLD = 0.35

# Video processing optimizations (CPU-friendly)
FRAME_SKIP = 3  # Process every 3rd frame to achieve fast processing without missing vehicles
DUPLICATE_DEBOUNCE_SECONDS = 1.8  # Debounce interval to avoid duplicate evidence for same vehicle

# COCO vehicle class mappings used by genuine YOLOv8
# 2: car, 3: motorcycle, 5: bus, 7: truck
COCO_VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}
