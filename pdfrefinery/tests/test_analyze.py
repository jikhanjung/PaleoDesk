import os
import json

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from PyQt6.QtWidgets import QApplication, QStyle, QIcon
from PyQt6.QtCore import QSettings

import pytest

from pdfrefinery import PDFModels
from pdfrefinery.PDFRefinery import MainWindow

# utility to init temp db
def init_db(path):
    PDFModels.db.init(path)
    PDFModels.db.connect()
    PDFModels.db.create_tables([
        PDFModels.PDFDocument,
        PDFModels.PageAnalysis,
        PDFModels.SessionData,
        PDFModels.StructuredElement,
        PDFModels.PrFigure,
    ])

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

def test_analyze_populates_structured_elements(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    db_path = tmp_path / 'db.sqlite'
    init_db(db_path)

    # patch service url
    settings = QSettings('PaleoBytes', 'PDFRefinery')
    settings.setValue('service/url', 'http://dummy')

    # patch icon function
    monkeypatch.setattr('pdfrefinery.PDFRefinery.get_analysis_done_icon', lambda: QIcon())

    # mock requests.post
    results = [{
        'page_number': 1,
        'type': 'text',
        'text': 'hello',
        'left': 0,
        'top': 0,
        'width': 100,
        'height': 10,
        'page_width': 100,
        'page_height': 200
    }]

    def fake_post(url, files=None, *a, **k):
        class Resp:
            status_code = 200
            def json(self_self):
                return results
            content = b''
        return Resp()

    monkeypatch.setattr('pdfrefinery.PDFRefinery.requests.post', fake_post)

    pdf_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Schwimmer 2012 page1.pdf')

    doc = PDFModels.PDFDocument.create(file_path=pdf_path, title='Sample', page_count=1)

    dummy = type('Dummy', (), {})()
    dummy.document_record = doc
    dummy.pdf_document = DummyDoc()
    dummy.document_data = {'page_structures': {}, 'initial_page_structures': {}, 'metadata': {}}
    dummy.pdf_viewer = type('V', (), {'current_page': 0})()
    dummy.status_label = DummyStatus()
    dummy.items_tree = DummyItemsTree()
    dummy.current_page = 0
    dummy.current_zotero_key = None
    dummy.update_page_display = lambda: None
    dummy.save_session = lambda: None
    dummy.ensure_normal_cursor = lambda: None

    result = MainWindow.analyze_pdf(dummy, pdf_path, force_analysis=True)
    assert result is True
    assert PDFModels.StructuredElement.select().count() == len(results)
    PDFModels.db.close()
