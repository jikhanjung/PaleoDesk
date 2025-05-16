from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, 
                           QLabel, QLineEdit, QComboBox, QDialogButtonBox, 
                           QTextEdit, QSizePolicy, QLayout)
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QImage, QPixmap
from PDFCommons import *
import logging
import fitz

# Get logger
logger = logging.getLogger('PrDialogs')


class PreferencesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings(COMPANY_NAME, PROGRAM_NAME)
        self.init_ui()
        self.load_settings()
        logger.debug("PreferencesDialog initialized")
        
    def init_ui(self):
        self.setWindowTitle("Preferences")
        layout = QVBoxLayout()
        
        # Analyze URL
        url_layout = QHBoxLayout()
        url_label = QLabel("Analyze URL:")
        self.url_edit = QLineEdit()
        url_layout.addWidget(url_label)
        url_layout.addWidget(self.url_edit)
        layout.addLayout(url_layout)
        
        # Logging Level
        log_layout = QHBoxLayout()
        log_label = QLabel("Logging Level:")
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        log_layout.addWidget(log_label)
        log_layout.addWidget(self.log_level_combo)
        layout.addLayout(log_layout)
        
        # Add some spacing
        layout.addSpacing(10)
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        
    def load_settings(self):
        self.url_edit.setText(self.settings.value("service/url", ""))
        level = self.settings.value("logging/level", "INFO")
        index = self.log_level_combo.findText(level)
        if index >= 0:
            self.log_level_combo.setCurrentIndex(index)
        logger.debug("Loaded settings from QSettings")
        
    def save_settings(self):
        self.settings.setValue("service/url", self.url_edit.text())
        new_level = self.log_level_combo.currentText()
        old_level = self.settings.value("logging/level", "INFO")
        self.settings.setValue("logging/level", new_level)
        
        # Update logging level if changed
        if new_level != old_level:
            level = getattr(logging, new_level)
            logger.setLevel(level)
            for handler in logger.handlers:
                handler.setLevel(level)
            logger.info(f"Logging level changed from {old_level} to {new_level}")
        else:
            logger.info(f"Saved service URL: {self.url_edit.text()}")
        
    def accept(self):
        self.save_settings()
        super().accept()

class ElementInfoDialog(QDialog):
    def __init__(self, element_data, main_window=None):
        super().__init__()
        self.element_data = element_data
        self.main_window = main_window
        self.category = None
        self.figure_element = None
        self.caption_element = None
        self.text_elements = []
        self._handle_element_data()
        #logger.info(f"element_data: {self.element_data}")
        self.init_ui()

    def _get_pixmap(self, elem):
        """Get or create a cached pixmap for an item"""
        # Create a unique cache key using page number, id, and coordinates
            
        try:
            coords = elem.get('coordinates', [])
            page = self.main_window.pdf_document[elem['page_number'] ]  # Convert to 0-based index
            zoom = 2  # Reduced zoom for better performance
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix)
            
            # Convert to QImage
            img = QImage(pix.samples, pix.width, pix.height, 
                       pix.stride, QImage.Format.Format_RGB888)
            
            # Get coordinates
            if coords and len(coords) >= 4:
                # Convert relative coordinates to absolute
                x1 = int(coords[0]['x'] * pix.width)
                y1 = int(coords[0]['y'] * pix.height)
                x2 = int(coords[2]['x'] * pix.width)
                y2 = int(coords[2]['y'] * pix.height)
                
                # Clip the region
                clipped_img = img.copy(x1, y1, x2 - x1, y2 - y1)
                
                # Create QPixmap and scale to reasonable size
                pixmap = QPixmap.fromImage(clipped_img)
                max_size = 400
                pixmap = pixmap.scaled(max_size, max_size, 
                                     Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
                
                # Cache the pixmap
                return pixmap
        except Exception as e:
            logger.error(f"Error creating image for item on page {elem['page_number']} {elem['id']}: {str(e)}")
        
        return None

    def _handle_element_data(self):
        """Handle element data for a given page number and element ID"""
        all_same_category = True
        category_set = set()
        for elem in self.element_data:
            category_set.add(elem['category'])
            if len(category_set) > 1:
                all_same_category = False
        
        self.category = 'text'
        if all_same_category:
            self.category = category_set.pop()
            #self.text_elements = self.element_data
        else:
            for type in category_set:
                if type.startswith('figure:'):
                    self.category = 'figure'
                    self.figure_element = None
                    self.caption_element = None
                    for elem in self.element_data:
                        if elem['category'].startswith('figure:'):
                            self.figure_element = elem
                        elif elem['category'].startswith('caption'):
                            self.caption_element = elem

    def _get_element_info(self, page_num, element_id):
        """Get element info for a given page number and element ID"""
        logger.info(f"Getting element info for page {type(page_num)} {page_num} and element {type(element_id)} {element_id}")
        for elem in self.element_data:
            if elem['page_number'] == page_num and elem['id'] == element_id:
                return elem
        return None
        
    def init_ui(self):
        self.setWindowTitle("Element Information")
        layout = QVBoxLayout()
        
        # Create form layout for basic information
        form_layout = QFormLayout()
        form_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
        #logger.info(f"element_data: {self.element_data}")
        
        # Add type and page information
        form_layout.addRow("Category:", QLabel(self.category.capitalize()))
        for elem in self.element_data:
            form_layout.addRow("ID:", QLabel(f"{elem.get('page_number', '')+1}-{int(elem.get('id', ''))+1}"))
        layout.addLayout(form_layout)

        caption_text_value = None
        figure_pixmap = None
        # If figure, try to get cropped image and caption
        if self.category == 'figure':
            # Try to get pixmap (cropped image)
            # Show figure image
            figure_pixmap = self._get_pixmap(self.figure_element)
            if figure_pixmap:
                image_label = QLabel()
                image_label.setPixmap(figure_pixmap)
                image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                image_label.setMinimumSize(200, 200)
                layout.addWidget(image_label, 1)
            # Show caption if found
            logger.info(f"caption_element: {self.caption_element}")
            if self.caption_element:
                caption_label = QLabel("Caption:")
                caption_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
                layout.addWidget(caption_label)
                caption_text = QTextEdit()
                caption_text.setReadOnly(True)
                caption_text.setPlainText(self.caption_element['content']['text'])
                caption_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                layout.addWidget(caption_text, 1)
        else:
            # Add caption in a read-only text area
            text_label = QLabel("Text:")
            text_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            layout.addWidget(text_label)
            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            text = ''
            for elem in self.element_data:
                if 'content' in elem:
                    if 'text' in elem['content']:
                        text += elem['content']['text']
            text_edit.setPlainText(text)
            text_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            layout.addWidget(text_edit, 1)
        # Add close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        button_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout.addWidget(button_box)
        self.setLayout(layout)
        self.setMinimumSize(600, 400)        