import os
import tempfile
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# set headless before Qt imports
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QSettings

from pdfrefinery import PDFModels

class TestPDFModelsCRUD(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory and database
        self.test_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.test_dir.name, 'test.db')
        PDFModels.db.init(self.db_path)
        PDFModels.db.connect()
        PDFModels.db.create_tables([
            PDFModels.PDFDocument,
            PDFModels.PageAnalysis,
            PDFModels.SessionData,
            PDFModels.StructuredElement,
            PDFModels.PrFigure,
        ])

    def tearDown(self):
        PDFModels.db.close()
        self.test_dir.cleanup()

    def test_crud(self):
        # create
        doc = PDFModels.PDFDocument.create(
            file_path='sample.pdf',
            file_hash='abc',
            title='Test',
            page_count=1
        )
        # read
        fetched = PDFModels.PDFDocument.get(PDFModels.PDFDocument.id == doc.id)
        self.assertEqual(fetched.title, 'Test')
        # update
        fetched.title = 'Changed'
        fetched.save()
        self.assertEqual(PDFModels.PDFDocument.get_by_id(doc.id).title, 'Changed')
        # delete
        fetched.delete_instance()
        self.assertEqual(PDFModels.PDFDocument.select().count(), 0)

if __name__ == "__main__":
    unittest.main()
