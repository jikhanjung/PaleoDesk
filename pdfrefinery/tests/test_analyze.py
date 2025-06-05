import sys
import unittest
from unittest.mock import patch
import os
import logging
import fitz  # PyMuPDF
import tempfile
from peewee import SqliteDatabase

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Static import for PDFModels
from PDFModels import *

# Setup in-memory DB and bind before importing models
test_db = SqliteDatabase(':memory:')

# Set each model's _meta.database to test_db
PDFDocument._meta.database = test_db
PageAnalysis._meta.database = test_db
SessionData._meta.database = test_db
StructuredElement._meta.database = test_db
PrFigure._meta.database = test_db

test_db.bind([
    PDFDocument,
    PageAnalysis,
    SessionData,
    StructuredElement,
    PrFigure,
], bind_refs=False, bind_backrefs=False)
test_db.create_tables([
    PDFDocument,
    PageAnalysis,
    SessionData,
    StructuredElement,
    PrFigure,
], safe=False)

# Print the schema of sessiondata table for debugging
print("SessionData table columns in test DB:")
for row in test_db.execute_sql("PRAGMA table_info(sessiondata);"):
    print(row)

from PyQt6.QtWidgets import QApplication, QStyle
from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QIcon
# Now import MainWindow (after models are bound)
from pdfrefinery.PDFRefinery import MainWindow

# Set all loggers to DEBUG for testing
logging.basicConfig(level=logging.DEBUG)
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.DEBUG)

class DummyPage:
    def get_text(self):
        return 'x' * 200

class DummyDoc:
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        return DummyPage()

class DummyStatus:
    def showMessage(self, *a, **k):
        pass

class DummyItem:
    def setIcon(self, col, icon):
        self.icon = icon

class DummyItemsTree:
    def __init__(self):
        self._item = DummyItem()
    def currentItem(self):
        return self._item

class TestAnalyzePDF(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        # Set the real service URL
        settings = QSettings('PaleoBytes', 'PDFRefinery')
        settings.setValue('service/url', 'http://localhost:8051')

    def test_analyze_real_pdf(self):
        # Create a real PDF with text in a temporary file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
            pdf_path = tmp_pdf.name

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Hello, PDFRefinery!", fontsize=20)
        doc.save(pdf_path)
        doc.close()

        # Create DB record
        db_doc = PDFDocument.create(file_path=pdf_path, title='Sample', page_count=1)

        # Create a real MainWindow instance
        main_window = MainWindow()
        main_window.document_record = db_doc
        main_window.pdf_document = fitz.open(pdf_path)
        main_window.document_data = {'page_structures': {}, 'initial_page_structures': {}, 'metadata': {}}
        main_window.pdf_viewer = type('V', (), {'current_page': 0})()
        main_window.status_label = DummyStatus()
        main_window.items_tree = DummyItemsTree()
        main_window.current_page = 0
        main_window.current_zotero_key = None
        main_window.current_file_path = pdf_path
        main_window.update_page_display = lambda: None
        main_window.ensure_normal_cursor = lambda: None

        result = MainWindow.analyze_pdf(main_window, pdf_path, force_analysis=True)
        self.assertTrue(result)
        # There should be at least one StructuredElement (for the text)
        self.assertGreater(StructuredElement.select().count(), 0)

        # Clean up
        main_window.pdf_document.close()
        os.remove(pdf_path)

if __name__ == "__main__":
    unittest.main()
