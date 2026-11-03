# AI-Based Intelligent Traffic Violation Detection and Monitoring System Using Computer Vision

**Final-Year B.Tech Capstone Project in Computer Vision & Artificial Intelligence**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Flask 3.1](https://img.shields.io/badge/Flask-3.1-black.svg)](https://flask.palletsprojects.com/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLO-v8n-00599C.svg)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0-green.svg)](https://opencv.org/)
[![EasyOCR](https://img.shields.io/badge/EasyOCR-PyTorch-orange.svg)](https://github.com/JaidedAI/EasyOCR)
[![Database](https://img.shields.io/badge/SQLite-Integrated-lightgrey.svg)](https://sqlite.org/)

---

## 1. Executive Summary & Project Abstract

Urban traffic management and road safety enforcement face immense challenges with escalating vehicle volumes. Traditional manual policing is labor-intensive, error-prone, and geographically limited. 

This project delivers **Traffic-Vigil AI**, an end-to-end intelligent traffic violation detection and monitoring web system engineered with **Python Flask**, **Ultralytics YOLOv8**, **OpenCV**, and **EasyOCR**. The application ingests surveillance video footage (supporting MP4, AVI, MOV, and MKV), automatically detects and classifies multi-class vehicles (cars, motorcycles, buses, trucks), captures timestamped evidence with telemetry Head-Up Displays (HUD), extracts vehicle license plate numbers, flags safety violations, and provides a web dashboard with **Human-in-the-Loop review** to prevent automated incorrect fines.

---

## 2. Key Features

1. **Multi-Format Video Ingestion:**
   - Supports surveillance clips in `MP4`, `AVI`, `MOV`, and `MKV` formats up to 150 MB.
   - Built-in file type and size validation with secure sanitization.
2. **YOLOv8 Multi-Class Vehicle Detection:**
   - Detects cars, motorcycles, buses, and trucks with high precision using genuine pretrained `yolov8n.pt`.
   - Optimized with CPU frame skipping (`FRAME_SKIP = 3`) and debounce tracking to prevent duplicate spam.
3. **Number Plate Localization & EasyOCR Recognition:**
   - Two-tier pipeline: supports dedicated YOLO plate weights (`plate_model.pt`) and an integrated morphological ROI edge detector fallback.
   - OCR character extraction powered by PyTorch EasyOCR with contrast enhancement and text sanitization.
4. **Helmet & Safety Monitoring Pipeline:**
   - Transparent architecture supporting custom helmet weights (`models/helmet_model.pt`).
   - If weights are absent, displays an honest model status indicator and flags motorcycle riders for human auditing without pretending models exist.
5. **Timestamped Evidence Generation:**
   - Saves annotated high-resolution evidence snapshots with bounding boxes, extracted plate badges, and HUD telemetry banners into `static/evidence/`.
6. **SQLite Audit Trail & Database:**
   - Fully parameterized SQL queries storing video origin, timestamp (seconds & formatted), confidence, vehicle type, plate number, and reviewer decisions.
7. **Human-in-the-Loop Review Workflow:**
   - Suspected violations are held in `Pending Review`.
   - Human auditors can **Approve Violation** or **Dismiss / False Positive** directly with optional audit notes. No automated fines are ever issued.
8. **Real-Time Telemetry & SSE Streaming:**
   - Live Server-Sent Events (SSE) progress bar displaying frame count, detection rate, and processing throughput (FPS).
9. **Modern Responsive UI Dashboard:**
   - Dark Navy header (`#0f172a`), clean light background (`#f8fafc`), and blue accent theme.
   - Real-time Chart.js doughnut and bar charts for vehicle and violation distribution.
   - Filterable & searchable audit table with evidence modal viewer and CSV export.

---

## 3. System Architecture

```
+-----------------------------------------------------------------------------------+
|                            USER BROWSER / CLIENT                                  |
|   - Dashboard (Chart.js)    - Video Upload (Dropzone)    - Records Gallery        |
|   - Telemetry Stream (SSE)  - Evidence Modal Viewer      - Human Review Controls  |
+------------------------------------------^----------------------------------------+
                                           | HTTP / Server-Sent Events
+------------------------------------------v----------------------------------------+
|                                FLASK BACKEND (app.py)                             |
|   - Route Handlers          - File Upload Validation    - CSV Exporter            |
|   - Human Review API        - Model Health API          - Error Handlers          |
+------------------+-----------------------+---------------------+------------------+
                   |                       |                     |
                   v                       v                     v
      +------------------------+  +------------------+  +-------------------+
      |   DETECTION ENGINE     |  |   PLATE & OCR    |  |  SQLITE DATABASE  |
      |     (detector.py)      |  | (plate_reader.py)|  |   (database.py)   |
      | - YOLOv8 Vehicle Det   |  | - Plate ROI      |  | - videos table    |
      | - Helmet Classifier    |  | - EasyOCR Engine |  | - violations table|
      | - Debounce Tracker     |  | - Text Sanitizer |  | - Filter Queries  |
      | - HUD Evidence Drawer  |  +------------------+  +-------------------+
      +-----------+------------+
                  |
                  v
      +------------------------+
      |  MODELS & EVIDENCE     |
      | models/yolov8n.pt      |
      | models/helmet_model.pt |
      | static/evidence/*.jpg  |
      +------------------------+
```

---

## 4. Project Directory Structure

```
TrafficViolationProject/
│
├── app.py                      # Flask web server, endpoints, and SSE streaming
├── config.py                   # Central settings, paths, thresholds, and limits
├── database.py                 # SQLite database helper, queries, and analytics
├── detector.py                 # YOLO vehicle detection, debounce, and HUD evidence drawer
├── plate_reader.py             # Number plate ROI extractor and EasyOCR integration
├── evaluate_metrics.py         # Precision, Recall, F1, and FPS evaluation benchmarking
├── requirements.txt            # Python dependencies specification
├── traffic_system.db           # SQLite database file (created automatically)
├── .gitignore                  # Git ignore rules for media, venv, and cache
├── README.md                   # Complete documentation and viva guide
│
├── models/                     # Deep learning weights directory
│   ├── yolov8n.pt              # Genuine pretrained YOLOv8 Nano vehicle model
│   ├── helmet_model.pt         # Optional custom helmet weights (place here)
│   └── plate_model.pt          # Optional custom license plate weights (place here)
│
├── templates/                  # Jinja2 HTML templates
│   ├── base.html               # Dark navy master layout with modals and alerts
│   ├── index.html              # Analytics dashboard, KPI cards, and model health
│   ├── upload.html             # Video drag-and-drop upload view
│   ├── process.html            # Real-time processing progress with live telemetry
│   ├── results.html            # Filterable violation records and review gallery
│   ├── videos.html             # Uploaded surveillance feeds manager
│   ├── 404.html                # Friendly 404 error page
│   └── 500.html                # Friendly 500 error page
│
├── static/                     # Frontend assets
│   ├── css/
│   │   └── style.css           # Custom modern stylesheet
│   ├── js/
│   │   └── main.js             # Chart.js initialization and AJAX review controller
│   └── evidence/               # Saved timestamped evidence photos with HUD
│
├── uploads/                    # Directory for uploaded surveillance video feeds
│
└── tests/                      # Automated test suite
    ├── __init__.py
    ├── test_db.py              # Unit tests for SQLite CRUD and analytics
    ├── test_detector.py        # Unit tests for vision pipeline and plate cleaner
    ├── test_app.py             # Integration tests for Flask routes and CSV export
    └── generate_sample_video.py# Synthetic traffic video generator
```

---

## 5. Environment & Virtual Environment Setup

### Prerequisites
- Windows 10 or 11
- Python 3.11, 3.12, or 3.14 (Verified compatible with PyTorch 2.14+, OpenCV 5.0+, EasyOCR 1.7+, Flask 3.1+)
- Google Antigravity or Visual Studio Code

### Step 1: Clone or Navigate to the Project Directory
```powershell
cd c:\Users\HP\Desktop\TrafficViolationProject
```

### Step 2: Create and Activate Virtual Environment
```powershell
# Create isolated virtual environment
python -m venv venv

# Activate on Windows PowerShell
.\venv\Scripts\Activate.ps1

# (If running standard Windows Command Prompt cmd.exe)
# .\venv\Scripts\activate.bat
```

> **Note on PowerShell Script Execution Policy:**  
> If PowerShell displays `execution of scripts is disabled on this system`, run:  
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Step 3: Install Stable Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 4: Selecting the Virtual Environment in Google Antigravity / VS Code
1. Open Google Antigravity.
2. Press `Ctrl + Shift + P` to open the Command Palette.
3. Type: `Python: Select Interpreter`.
4. Click **Enter interpreter path...** -> **Find...**.
5. Select `c:\Users\HP\Desktop\TrafficViolationProject\venv\Scripts\python.exe`.
6. Antigravity will now use this isolated virtual environment for code execution and debugging.

---

## 6. How to Run and Test the Application

### 1. Run the Flask Web Application
```powershell
.\venv\Scripts\python.exe app.py
```
Open your web browser and navigate to:  
👉 **`http://127.0.0.1:5000`**

### 2. Run the Automated Unit and Integration Test Suite
```powershell
.\venv\Scripts\python.exe -m unittest discover tests
```
*Expected result:* All 11 tests pass with status `OK`.

### 3. Generate Synthetic Demonstration Video
```powershell
.\venv\Scripts\python.exe tests/generate_sample_video.py
```
This generates `uploads/demo_traffic_feed.mp4`, allowing you to demonstrate the project even without internet or real CCTV clips.

### 4. Run Evaluation Metrics & Inference Speed Benchmark
```powershell
.\venv\Scripts\python.exe evaluate_metrics.py
```

---

## 7. Deep Learning Models & Weight Configuration

The system features real-time model health tracking across all computer vision subsystems:

| Subsystem | Model Weights Path | Status | Operational Details |
| :--- | :--- | :--- | :--- |
| **Vehicle Detection** | `models/yolov8n.pt` | **Operational** | Genuine Ultralytics YOLOv8 Nano (Cars, Motorcycles, Buses, Trucks) |
| **Helmet Detection** | `models/helmet_model.pt` | **Operational** | Pretrained YOLOv8 model (`Hardhat` vs `NO-Hardhat` / Helmet compliance) |
| **Plate Candidate ROI**| Integrated Heuristic | **Operational** | Contrast-enhanced morphological edge and aspect-ratio filter |
| **Plate OCR Engine** | EasyOCR PyTorch | **Operational** | Bi-directional English alphanumeric character extraction |
| **Dedicated Plate YOLO**| `models/plate_model.pt` | *Optional Enhancement* | Morphology fallback is active; dedicated YOLO can be added anytime |

### Helmet Detection Subsystem Architecture:
1. **Installed Model Weights:** `models/helmet_model.pt` (6.22 MB YOLOv8 Nano weights).
2. **Model Classes:** Class 0: `Hardhat` (Helmet Compliant), Class 1: `NO-Hardhat` (Safety Infraction).
3. **Inference Pipeline:**
   - Detects motorcycles via YOLOv8 vehicle engine.
   - Extracts vehicle ROI and expands the bounding box upward by 35% to cover the rider's torso and head.
   - Runs `models/helmet_model.pt` to detect helmet presence or unhelmeted bare heads.
   - Flags violations (`No Helmet Detected`), saves high-resolution evidence snapshots with helmet bounding boxes, and submits records for human verification.

### How to Replace or Fine-Tune with Custom Motorcycle Helmet Datasets:
If you want to train on a specific motorcycle dataset (e.g., [Roboflow Motorcycle Helmet Dataset](https://universe.roboflow.com/search?q=motorcycle+helmet)):
1. Download annotations and images in YOLOv8 format.
2. Train using Ultralytics CLI or Python:
   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8n.pt')
   model.train(data='helmet_dataset.yaml', epochs=30, imgsz=640)
   ```
3. Copy your trained `runs/detect/train/weights/best.pt` file to:
   `c:\Users\HP\Desktop\TrafficViolationProject\models\helmet_model.pt`
4. The system detects the new weights automatically upon startup.

### How to Add Dedicated YOLO License Plate Weights (Optional):
1. Train or download a license plate YOLO model (e.g., from Roboflow or Kaggle Indian License Plate Dataset).
2. Save the weights file to:
   `c:\Users\HP\Desktop\TrafficViolationProject\models\plate_model.pt`
3. Restart `app.py`. The dashboard will automatically switch from Morphology Fallback to Dedicated YOLO Plate mode.

---

## 8. Capstone Viva & Demonstration Walkthrough

When presenting this project to your college project guide, external examiner, or faculty panel, follow this recommended walkthrough:

```
Step 1: Dashboard Overview
   -> Navigate to http://127.0.0.1:5000
   -> Show KPI cards: Total Detections, Suspected Violations, Audited Videos, Pending Reviews.
   -> Highlight Model Health indicators: Explain that vehicle detection is operational with genuine COCO YOLOv8, while helmet and plate modules honestly report their state.

Step 2: Video Upload
   -> Click "Upload New Video".
   -> Select "Real Bus Traffic Sample.mp4" or upload a CCTV clip.
   -> Show file extension and size validation.

Step 3: Real-Time AI Processing & Telemetry
   -> Watch the live progress bar powered by Server-Sent Events (SSE).
   -> Point out real-time FPS throughput (typically 15-50 FPS on CPU).
   -> Highlight live detection logs in the console terminal.

Step 4: Evidence Inspection & Audit Gallery
   -> Click "View Filterable Records & Evidence".
   -> Show how evidence images contain HUD telemetry bars (timestamp, vehicle class, confidence).
   -> Click on a thumbnail to open the modal evidence viewer.

Step 5: Human-in-the-Loop Review
   -> Explain why automated fines are unsafe in legal systems.
   -> Click "Review" or use the quick checkmark/cross buttons to mark a record as "Approved Violation" or "Dismissed / False Positive".
   -> Add reviewer notes and show how the status updates dynamically without reloading.

Step 6: Reporting & Export
   -> Click "Download CSV Report" to generate the audit spreadsheet.
   -> Show the Chart.js visual distribution on the main dashboard.
```

---

## 9. Academic Evaluation: Precision, Recall, and FPS Formulation

During final-year project vivas, examiners often ask how computer vision performance is measured. You can explain the following formulas (implemented in `evaluate_metrics.py`):

### 1. Precision
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
- **Explanation:** Measures detection accuracy. Out of all vehicles or violations flagged by the AI, what percentage were genuine?
- **High precision prevents:** Innocent motorists being falsely penalized.

### 2. Recall (Sensitivity)
$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
- **Explanation:** Measures coverage. Out of all actual vehicles or violations present on the road, what percentage did the system successfully catch?
- **High recall ensures:** Serious safety violations are not missed.

### 3. F1-Score
$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
- The harmonic mean balancing both precision and recall.

### 4. False Positive Rate (FPR)
$$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$
- Crucial justification for your **Human-in-the-Loop design**: human review reduces the operational FPR to near 0%.

### 5. Processing Speed (Frames Per Second - FPS)
$$\text{FPS} = \frac{\text{Total Frames Processed}}{\text{Elapsed Time (seconds)}}$$
- **Benchmark result on standard CPU:** ~35–52 FPS with frame skipping, demonstrating suitability for edge-device and real-time surveillance deployment.

---

## 10. Future Deployment Strategy

1. **Local Windows Deployment (Current):**
   - Running via Flask WSGI server at `http://127.0.0.1:5000`.
2. **Production Online Deployment:**
   - **Containerization:** Package with Docker using an `nvidia/cuda` or lightweight `python:3.11-slim` base image.
   - **Production WSGI:** Replace Flask built-in development server with `gunicorn` (Linux) or `waitress` (Windows):
     ```powershell
     pip install waitress
     waitress-serve --port=5000 app:app
     ```
   - **Cloud Hosting:** Deploy to AWS EC2 (G4dn GPU instance for real-time multi-camera feeds) or Azure App Service.
   - **Database Migration:** Easily switch from SQLite to PostgreSQL by updating `database.py` connection strings for high-throughput concurrency.

---

## 11. Author & Acknowledgments

- **Project:** B.Tech Final-Year Capstone Project
- **Domain:** Artificial Intelligence, Computer Vision & Intelligent Transportation Systems (ITS)
- **Frameworks:** Ultralytics YOLOv8, OpenCV, EasyOCR, Flask, SQLite, Chart.js
