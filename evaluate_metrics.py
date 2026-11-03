"""
Performance & Evaluation Benchmarking Script
AI-Based Intelligent Traffic Violation Detection and Monitoring System
B.Tech Capstone Project Viva & Research Evaluation Utility
"""

import time
import os
import cv2
import numpy as np
from detector import get_vehicle_model, get_system_model_status
from config import VEHICLE_MODEL_PATH, FRAME_SKIP

def calculate_metrics(tp, fp, fn):
    """
    Calculate Precision, Recall, and F1-Score.
    
    Formulas:
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1-Score  = 2 * (Precision * Recall) / (Precision + Recall)
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "True Positives (TP)": tp,
        "False Positives (FP)": fp,
        "False Negatives (FN)": fn,
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1-Score": round(f1 * 100, 2)
    }

def benchmark_fps(video_path, max_frames=100):
    """
    Measures processing speed and throughput in Frames Per Second (FPS).
    """
    if not os.path.exists(video_path):
        print(f"[ERROR] Video path does not exist: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return None

    model = get_vehicle_model()
    frame_count = 0
    inference_times = []

    print(f"\n[BENCHMARK] Running inference speed test on: {os.path.basename(video_path)}")
    print(f"[BENCHMARK] Testing with frame skip = {FRAME_SKIP}...")

    total_start = time.time()

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        start_inf = time.time()
        _ = model(frame, conf=0.40, verbose=False)
        inf_duration = time.time() - start_inf
        inference_times.append(inf_duration)

    total_duration = time.time() - total_start
    cap.release()

    if not inference_times:
        print("[ERROR] No frames processed.")
        return None

    processed_count = len(inference_times)
    avg_inf_ms = (sum(inference_times) / processed_count) * 1000.0
    overall_fps = frame_count / total_duration if total_duration > 0 else 0.0

    print("\n" + "="*50)
    print("      COMPUTER VISION INFERENCE BENCHMARK")
    print("="*50)
    print(f"Total Video Frames Inspected: {frame_count}")
    print(f"Frames Passed Through Model:  {processed_count}")
    print(f"Average Inference Latency:    {avg_inf_ms:.2f} ms per frame")
    print(f"System Throughput (FPS):      {overall_fps:.2f} FPS")
    print("="*50)

    return {
        "frames_tested": frame_count,
        "avg_latency_ms": round(avg_inf_ms, 2),
        "throughput_fps": round(overall_fps, 2)
    }

def generate_evaluation_report():
    """Generates a structured viva explanation report for project evaluation."""
    print("\n" + "="*60)
    print("   AI TRAFFIC VIOLATION SYSTEM - EVALUATION REPORT")
    print("="*60)
    
    # Model Availability Report
    status = get_system_model_status()
    print("\n1. MODEL SUBSYSTEM STATUS:")
    for k, v in status.items():
        print(f"   * {v['name']}: {v['status']} (File: {v['file']})")

    # Sample Validation Metric Table (Example Ground Truth validation)
    print("\n2. SAMPLE DETECTION PERFORMANCE METRICS (Formula Demonstration):")
    # Example validation on 50 ground-truth annotated traffic frames:
    # 45 correctly detected vehicles (TP), 3 background false alarms (FP), 2 missed vehicles (FN)
    metrics = calculate_metrics(tp=45, fp=3, fn=2)
    for mk, mv in metrics.items():
        unit = "%" if ("Precision" in mk or "Recall" in mk or "F1" in mk) else ""
        print(f"   - {mk}: {mv}{unit}")

    print("\n3. HOW TO PRESENT THIS TO YOUR CAPSTONE EXAMINER:")
    print("   - Precision answers: 'Of all vehicles/violations flagged, how many were actually real?'")
    print("   - Recall answers:    'Of all real vehicles/violations on the road, how many did the system catch?'")
    print("   - F1-Score:          The harmonic mean balancing Precision and Recall.")
    print("   - FPS:               Processing speed in frames per second (proves real-time feasibility).")
    print("="*60)

if __name__ == '__main__':
    generate_evaluation_report()
    demo_video = os.path.join(os.path.dirname(__file__), 'uploads', 'demo_traffic_feed.mp4')
    if os.path.exists(demo_video):
        benchmark_fps(demo_video)
