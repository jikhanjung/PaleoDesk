import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem, QScrollArea
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QFont
from PyQt6.QtCore import Qt
import fitz  # PyMuPDF
from PIL import Image
import imagehash
import io

def overlay_phash_on_pixmap(pixmap, phash_str):
    painter = QPainter(pixmap)
    painter.setPen(QColor(255, 0, 0))
    font = QFont()
    font.setPointSize(max(10, pixmap.width() // 40))
    painter.setFont(font)
    w, h = pixmap.width(), pixmap.height()
    grid_w, grid_h = w / 8, h / 8
    # If phash_str is hex (16 chars), expand to 64 bits
    if len(phash_str) == 16:
        bits = bin(int(phash_str, 16))[2:].zfill(64)
        hex_chars = list(phash_str)
        overlay_bits = list(bits)
    elif len(phash_str) == 64:
        overlay_bits = list(phash_str)
        # Compose hex_chars from bits
        hex_chars = [format(int(''.join(overlay_bits[i*4:(i+1)*4]), 2), 'x') for i in range(16)]
    else:
        overlay_bits = list(phash_str.ljust(64))
        hex_chars = ['?'] * 16
    for i in range(64):
        row = i // 8
        col = i % 8
        x = int(col * grid_w + grid_w / 2 - font.pointSize())
        y = int(row * grid_h + grid_h / 2)
        # Draw bit (0/1) in red
        painter.setPen(QColor(255, 0, 0))
        painter.drawText(x, y, overlay_bits[i])
        # Draw hex digit in blue (one per 4 bits, so only once per 4th cell)
        if i % 4 == 0:
            hex_idx = i // 4
            hx = int(col * grid_w + grid_w / 2 - font.pointSize())
            hy = int(row * grid_h + grid_h / 2 + font.pointSize())
            painter.setPen(QColor(0, 0, 255))
            painter.drawText(hx, hy, hex_chars[hex_idx])
    painter.end()
    return pixmap

class PDFPHashPanel(QWidget):
    def __init__(self, side_label):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.open_button = QPushButton(f'Open PDF ({side_label})')
        self.layout.addWidget(self.open_button)

        self.list_widget = QListWidget()
        self.layout.addWidget(self.list_widget)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        self.layout.addWidget(self.scroll_area)

        self.pdf_doc = None
        self.page_images = []
        self.page_hashes = []

    def load_pdf(self, file_path):
        self.pdf_doc = fitz.open(file_path)
        self.page_images = []
        self.page_hashes = []
        self.list_widget.clear()
        for i, page in enumerate(self.pdf_doc):
            pix = page.get_pixmap(dpi=100)
            img_bytes = pix.tobytes("ppm")
            pil_img = Image.open(io.BytesIO(img_bytes))
            phash = imagehash.phash(pil_img)
            self.page_images.append(pil_img)
            self.page_hashes.append(str(phash))
            item = QListWidgetItem(f"Page {i+1}: pHash={phash}")
            self.list_widget.addItem(item)
        if self.page_images:
            self.list_widget.setCurrentRow(0)

    def display_page(self, row):
        if 0 <= row < len(self.page_images):
            pil_img = self.page_images[row]
            phash_str = self.page_hashes[row]
            data = pil_img.convert('RGBA').tobytes('raw', 'RGBA')
            qimg = QImage(data, pil_img.width, pil_img.height, QImage.Format.Format_RGBA8888)
            pixmap = QPixmap.fromImage(qimg)
            # Overlay pHash chars
            pixmap = overlay_phash_on_pixmap(pixmap, phash_str)
            self.image_label.setPixmap(pixmap)
        else:
            self.image_label.clear()

class PDFPHashCompareViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PDF Page pHash Compare Viewer')
        self.resize(1600, 800)
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        self.left_panel = PDFPHashPanel('Left')
        self.right_panel = PDFPHashPanel('Right')
        self.layout.addWidget(self.left_panel)
        self.layout.addWidget(self.right_panel)

        self.left_panel.open_button.clicked.connect(self.open_left_pdf)
        self.right_panel.open_button.clicked.connect(self.open_right_pdf)

        self.left_panel.list_widget.currentRowChanged.connect(self.sync_page_selection_left)
        self.right_panel.list_widget.currentRowChanged.connect(self.sync_page_selection_right)

        self._syncing = False

    def open_left_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open PDF (Left)', '', 'PDF Files (*.pdf)')
        if not file_path:
            return
        self.left_panel.load_pdf(file_path)
        # Optionally sync right panel to page 0
        if self.right_panel.page_images:
            self.right_panel.list_widget.setCurrentRow(0)

    def open_right_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open PDF (Right)', '', 'PDF Files (*.pdf)')
        if not file_path:
            return
        self.right_panel.load_pdf(file_path)
        # Optionally sync left panel to page 0
        if self.left_panel.page_images:
            self.left_panel.list_widget.setCurrentRow(0)

    def sync_page_selection_left(self, row):
        if self._syncing:
            return
        self.left_panel.display_page(row)
        # Sync right panel if possible
        if self.right_panel.page_images and 0 <= row < len(self.right_panel.page_images):
            self._syncing = True
            self.right_panel.list_widget.setCurrentRow(row)
            self._syncing = False
        else:
            self.right_panel.image_label.clear()

    def sync_page_selection_right(self, row):
        if self._syncing:
            return
        self.right_panel.display_page(row)
        # Sync left panel if possible
        if self.left_panel.page_images and 0 <= row < len(self.left_panel.page_images):
            self._syncing = True
            self.left_panel.list_widget.setCurrentRow(row)
            self._syncing = False
        else:
            self.left_panel.image_label.clear()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = PDFPHashCompareViewer()
    viewer.show()
    sys.exit(app.exec()) 