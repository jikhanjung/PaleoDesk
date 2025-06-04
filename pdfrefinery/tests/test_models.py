import os
import tempfile
import json

# set headless before Qt imports
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QSettings

import pytest

from pdfrefinery import PDFModels

# Helper to initialize temp database
def init_temp_db(tmp_path):
    db_path = os.path.join(tmp_path, 'test.db')
    PDFModels.db.init(db_path)
    PDFModels.db.connect()
    PDFModels.db.create_tables([
        PDFModels.PDFDocument,
        PDFModels.PageAnalysis,
        PDFModels.SessionData,
        PDFModels.StructuredElement,
        PDFModels.PrFigure,
    ])
    return db_path


def test_pdfmodels_crud(tmp_path):
    init_temp_db(tmp_path)
    # create
    doc = PDFModels.PDFDocument.create(
        file_path='sample.pdf',
        file_hash='abc',
        title='Test',
        page_count=1
    )
    # read
    fetched = PDFModels.PDFDocument.get(PDFModels.PDFDocument.id == doc.id)
    assert fetched.title == 'Test'
    # update
    fetched.title = 'Changed'
    fetched.save()
    assert PDFModels.PDFDocument.get_by_id(doc.id).title == 'Changed'
    # delete
    fetched.delete_instance()
    assert PDFModels.PDFDocument.select().count() == 0
    PDFModels.db.close()
