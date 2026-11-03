import os
import cv2
import numpy as np

def generate_sample_traffic_video(output_path, num_frames=120, fps=25):
    """
    Generates a synthetic traffic video with animated vehicles and license plates
    to verify and demonstrate the complete detection and OCR pipeline.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 640, 480
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[INFO] Generating synthetic traffic video at: {output_path}")

    for f in range(num_frames):
        # Base road canvas (Dark asphalt)
        frame = np.full((height, width, 3), (45, 45, 50), dtype=np.uint8)

        # Draw road borders (curbs)
        cv2.rectangle(frame, (0, 0), (width, 80), (70, 130, 70), -1)      # Green grass top
        cv2.rectangle(frame, (0, 400), (width, height), (70, 130, 70), -1)  # Green grass bottom
        cv2.line(frame, (0, 80), (width, 80), (220, 220, 220), 4)
        cv2.line(frame, (0, 400), (width, 400), (220, 220, 220), 4)

        # Center dashed yellow divider line
        offset = (f * 8) % 60
        for x in range(-60 + offset, width + 60, 60):
            cv2.line(frame, (x, 240), (x + 35, 240), (0, 215, 255), 3)

        # Lane markings (white dashed)
        for x in range(-60 + offset, width + 60, 60):
            cv2.line(frame, (x, 160), (x + 25, 160), (200, 200, 200), 2)
            cv2.line(frame, (x, 320), (x + 25, 320), (200, 200, 200), 2)

        # --- Vehicle 1: Car moving across the lane ---
        car_x = int(-180 + (f * 7))
        car_y = 120
        if -200 < car_x < width + 50:
            # Car Body (Sleek Blue Sedan)
            cv2.rectangle(frame, (car_x, car_y + 15), (car_x + 140, car_y + 70), (180, 50, 30), -1)
            # Car Cabin / Roof
            cv2.rectangle(frame, (car_x + 30, car_y), (car_x + 105, car_y + 35), (200, 70, 50), -1)
            # Windshields
            cv2.rectangle(frame, (car_x + 35, car_y + 5), (car_x + 65, car_y + 30), (240, 240, 240), -1)
            cv2.rectangle(frame, (car_x + 72, car_y + 5), (car_x + 100, car_y + 30), (240, 240, 240), -1)
            # Wheels
            cv2.circle(frame, (car_x + 28, car_y + 70), 14, (15, 15, 15), -1)
            cv2.circle(frame, (car_x + 112, car_y + 70), 14, (15, 15, 15), -1)

            # High-contrast License Plate at rear
            plate_x = car_x + 5
            plate_y = car_y + 40
            cv2.rectangle(frame, (plate_x, plate_y), (plate_x + 48, plate_y + 18), (255, 255, 255), -1)
            cv2.rectangle(frame, (plate_x, plate_y), (plate_x + 48, plate_y + 18), (0, 0, 0), 1)
            cv2.putText(frame, "DL01", (plate_x + 4, plate_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # --- Vehicle 2: Motorcycle with rider in lower lane ---
        bike_x = int(width + 80 - (f * 6))
        bike_y = 280
        if -100 < bike_x < width + 100:
            # Wheels
            cv2.circle(frame, (bike_x + 15, bike_y + 55), 14, (20, 20, 20), -1)
            cv2.circle(frame, (bike_x + 75, bike_y + 55), 14, (20, 20, 20), -1)
            # Bike Body
            cv2.rectangle(frame, (bike_x + 25, bike_y + 40), (bike_x + 65, bike_y + 55), (0, 165, 255), -1)
            # Rider Body
            cv2.circle(frame, (bike_x + 45, bike_y + 15), 12, (200, 160, 130), -1) # Rider Head
            cv2.line(frame, (bike_x + 45, bike_y + 25), (bike_x + 45, bike_y + 48), (50, 50, 50), 6) # Torso

        out.write(frame)

    out.release()
    print(f"[SUCCESS] Synthetic traffic video successfully generated ({num_frames} frames) -> {output_path}")

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target = os.path.join(project_root, 'uploads', 'demo_traffic_feed.mp4')
    generate_sample_traffic_video(target)
