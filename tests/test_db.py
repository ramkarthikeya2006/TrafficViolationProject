import os
import unittest
import database

class TestDatabase(unittest.TestCase):
    def setUp(self):
        # Initialize database
        database.init_db()

    def test_database_crud(self):
        # 1. Add video
        video_id = database.add_video("test_traffic.mp4", "Original Traffic.mp4", "uploads/test_traffic.mp4")
        self.assertIsNotNone(video_id)
        
        # 2. Retrieve video
        video = database.get_video(video_id)
        self.assertEqual(video['filename'], "test_traffic.mp4")
        self.assertEqual(video['status'], "pending")

        # 3. Update video status
        database.update_video_status(video_id, "completed", total_frames=120, processed_frames=120, duration=4.8)
        updated_vid = database.get_video(video_id)
        self.assertEqual(updated_vid['status'], "completed")
        self.assertEqual(updated_vid['total_frames'], 120)

        # 4. Add violation detection
        v_id = database.add_violation(
            video_id=video_id,
            video_name="Original Traffic.mp4",
            event_type="No Helmet Detected",
            vehicle_type="motorcycle",
            timestamp_sec=2.4,
            formatted_time="00:02.4",
            plate_text="KA05MJ1234",
            confidence=0.88,
            evidence_image="test_evidence.jpg",
            review_status="Pending Review"
        )
        self.assertIsNotNone(v_id)

        # 5. Fetch violation
        violation = database.get_violation(v_id)
        self.assertEqual(violation['plate_text'], "KA05MJ1234")
        self.assertEqual(violation['review_status'], "Pending Review")

        # 6. Update human review status
        updated = database.update_violation_review(v_id, "Approved Violation", "Verified absence of helmet")
        self.assertTrue(updated)
        violation_after = database.get_violation(v_id)
        self.assertEqual(violation_after['review_status'], "Approved Violation")
        self.assertEqual(violation_after['reviewer_notes'], "Verified absence of helmet")

        # 7. Check stats calculation
        stats = database.get_dashboard_stats()
        self.assertGreaterEqual(stats['total_detections'], 1)
        self.assertGreaterEqual(stats['approved_violations'], 1)

if __name__ == '__main__':
    unittest.main()
