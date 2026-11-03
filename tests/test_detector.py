import os
import unittest
import numpy as np
import cv2

import detector
import plate_reader
from config import VEHICLE_MODEL_PATH

class TestDetectorModule(unittest.TestCase):
    def test_model_status_reporting(self):
        status = detector.get_system_model_status()
        self.assertIn('vehicle_model', status)
        self.assertIn('helmet_model', status)
        self.assertIn('plate_model', status)
        
        # Vehicle model weights should be present
        self.assertTrue(status['vehicle_model']['available'])
        self.assertEqual(status['vehicle_model']['status'], 'Operational')

    def test_timestamp_formatting(self):
        self.assertEqual(detector.format_timestamp(0), "00:00.0")
        self.assertEqual(detector.format_timestamp(65.4), "01:05.4")
        self.assertEqual(detector.format_timestamp(125.0), "02:05.0")

    def test_hud_evidence_drawing(self):
        # Create a blank synthetic test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        vehicle_box = [100, 150, 400, 350]
        
        annotated = detector.draw_hud_and_evidence(
            frame=frame,
            vehicle_box=vehicle_box,
            vehicle_type='car',
            event_type='Vehicle Detected - Car',
            confidence=0.92,
            timestamp_str='00:03.5',
            plate_text='DL01AB1234',
            plate_box=(50, 120, 180, 160),
            is_violation=False
        )

        self.assertEqual(annotated.shape, frame.shape)
        # Check that frame is no longer pure black
        self.assertTrue(np.any(annotated > 0))

    def test_helmet_model_loading(self):
        self.assertTrue(detector.is_helmet_model_available())
        model = detector.get_helmet_model()
        self.assertIsNotNone(model)
        self.assertIn(0, model.names)
        status = detector.get_system_model_status()
        self.assertTrue(status['helmet_model']['available'])
        self.assertEqual(status['helmet_model']['status'], 'Operational')

    def test_hud_evidence_with_helmet_boxes(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        vehicle_box = [100, 150, 400, 350]
        helmet_boxes = [(20, 20, 80, 80, "NO HELMET", 0.88, False)]
        annotated = detector.draw_hud_and_evidence(
            frame=frame,
            vehicle_box=vehicle_box,
            vehicle_type='motorcycle',
            event_type='No Helmet Detected',
            confidence=0.88,
            timestamp_str='00:05.0',
            plate_text='N/A',
            is_violation=True,
            helmet_boxes=helmet_boxes,
            crop_offset=(100, 100)
        )
        self.assertEqual(annotated.shape, frame.shape)
        self.assertTrue(np.any(annotated > 0))

    def test_plate_text_cleaner(self):
        self.assertEqual(plate_reader.clean_plate_text("  MH 12 - DE 1433  "), "MH12DE1433")
        self.assertEqual(plate_reader.clean_plate_text("!#KA53..AB9999"), "KA53AB9999")
        self.assertEqual(plate_reader.clean_plate_text("x"), "N/A")

if __name__ == '__main__':
    unittest.main()
