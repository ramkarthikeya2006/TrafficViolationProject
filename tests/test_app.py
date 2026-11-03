import unittest
import json
from app import app
import database

class TestAppRoutes(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        database.init_db()

    def test_dashboard_route(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'TRAFFIC-VIGIL AI', response.data)

    def test_upload_page_get(self):
        response = self.client.get('/upload')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Upload Traffic Video', response.data)

    def test_records_route(self):
        response = self.client.get('/records')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Audit Records', response.data)

    def test_api_stats(self):
        response = self.client.get('/api/stats')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('total_detections', data)
        self.assertIn('suspected_violations', data)

    def test_api_model_status(self):
        response = self.client.get('/api/model_status')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('vehicle_model', data)
        self.assertTrue(data['vehicle_model']['available'])

    def test_export_csv(self):
        response = self.client.get('/export_csv')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'text/csv')

if __name__ == '__main__':
    unittest.main()
