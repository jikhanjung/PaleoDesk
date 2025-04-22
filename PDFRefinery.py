import sys
import logging
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QFileDialog, QToolBar,
                           QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                           QDialogButtonBox, QMenuBar, QMenu, QMessageBox,
                           QTreeWidget, QTreeWidgetItem, QDockWidget, QSplitter,
                           QComboBox, QRadioButton, QButtonGroup, QTabWidget,
                           QGridLayout, QStackedWidget, QFormLayout, QTextEdit,
                           QScrollArea, QSizePolicy, QLayout)
from PyQt6.QtCore import Qt, QPoint, QSettings
from PyQt6.QtGui import QImage, QPixmap, QPainter, QAction, QCursor, QIcon, QPen, QColor
import fitz  # PyMuPDF
import datetime
import os
import requests
import json
import sqlite3
import shutil
from peewee import DoesNotExist
from PDFModels import (db, PDFDocument, PageAnalysis, SessionData, init_database,
                      calculate_file_hash, DEFAULT_LOG_DIRECTORY, COMPANY_NAME,
                      PROGRAM_NAME, DEFAULT_DB_DIRECTORY, DB_PATH)
import hashlib

PROGRAM_VERSION = "0.0.1"
PROGRAM_AUTHOR = "Jikhan Jung"
PROGRAM_COPYRIGHT = "©2025 Jikhan Jung"

# Get user profile directory
USER_PROFILE_DIRECTORY = os.path.expanduser('~')

# Define directory structure
DEFAULT_STORAGE_DIRECTORY = os.path.join(DEFAULT_DB_DIRECTORY, "data/")
DB_BACKUP_DIRECTORY = os.path.join(DEFAULT_DB_DIRECTORY, "backups/")

# Create necessary directories
for directory in [DEFAULT_STORAGE_DIRECTORY, DB_BACKUP_DIRECTORY]:
    os.makedirs(directory, exist_ok=True)

# Get the path to the resource file
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Configure logging
def setup_logging():
    # Create logs directory if it doesn't exist
    if not os.path.exists(DEFAULT_LOG_DIRECTORY):
        os.makedirs(DEFAULT_LOG_DIRECTORY)
        
    # Create log filename with date
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(DEFAULT_LOG_DIRECTORY, f"{PROGRAM_NAME}_{today}.log")
    
    # Get logging level from settings
    settings = QSettings(COMPANY_NAME, PROGRAM_NAME)
    level_name = settings.value("logging/level", "INFO")
    level = getattr(logging, level_name, logging.INFO)
    
    # Configure logging with UTF-8 encoding
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            # Use a custom StreamHandler for console that handles Unicode
            logging.StreamHandler(sys.stdout)  # stdout is typically UTF-8 on modern systems
        ]
    )
    logger = logging.getLogger(PROGRAM_NAME)
    logger.info(f"Logging initialized at {level_name} level. Log file: {log_file}")
    return logger

# Initialize logger
logger = setup_logging()

def setup_icons():
    # Create icons directory if it doesn't exist
    icons_dir = "icons"
    if not os.path.exists(icons_dir):
        os.makedirs(icons_dir)
        logger.info(f"Created icons directory: {icons_dir}")
    
    # Check if icon file exists
    icon_path = os.path.join(icons_dir, f"{PROGRAM_NAME}.png")
    if not os.path.exists(icon_path):
        logger.warning(f"Icon file not found: {icon_path}")
        return None
    
    return QIcon(icon_path)

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

class PDFViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.doc = None
        self.current_page = 0
        self.zoom = 1.0
        self.pan_offset = QPoint(0, 0)
        self.last_pan_pos = None
        self.last_mouse_pos = None
        self.pixmap = None
        self.bounding_boxes = []
        self.show_boxes = False
        
        # Enable mouse tracking for panning
        self.setMouseTracking(True)
        logger.debug("PDFViewer initialized")
        
    def open_pdf(self, file_path):
        if file_path:
            try:
                self.doc = fitz.open(file_path)
                self.current_page = 0
                self.pan_offset = QPoint(0, 0)
                self.display_page()
                logger.info(f"Opened PDF file: {file_path}")
            except Exception as e:
                logger.error(f"Error opening PDF file {file_path}: {str(e)}")
                raise
                
    def set_bounding_boxes(self, boxes):
        """Set bounding boxes for the current page"""
        self.bounding_boxes = boxes
        if boxes:
            # Log box information for debugging
            for i, box in enumerate(boxes):
                logger.debug(f"Box {i}: pos=({box['coordinates'][0]['x']:.3f}, {box['coordinates'][0]['y']:.3f}), " +
                           f"type={box.get('category', 'unknown')}")
        logger.info(f"Set {len(boxes)} bounding boxes")
        self.update()
        
    def toggle_bounding_boxes(self, show):
        """Toggle bounding box display"""
        self.show_boxes = show
        self.update()
        
    def paintEvent(self, event):
        if self.pixmap:
            painter = QPainter(self)
            painter.drawPixmap(self.pan_offset, self.pixmap)
            
            # Draw bounding boxes if enabled
            if self.show_boxes and self.bounding_boxes and self.doc:
                # Define colors for different categories with alpha values
                category_colors = {
                    'page header': QColor(0, 0, 139, 64),  # darkBlue with alpha
                    'title': QColor(0, 100, 0, 64),        # darkGreen with alpha
                    'section header': QColor(0, 128, 0, 64),  # green with alpha
                    'text': QColor(0, 0, 255, 64),         # blue with alpha
                    'picture': QColor(255, 255, 0, 64),    # yellow with alpha
                    'caption': QColor(139, 139, 0, 64),    # darkYellow with alpha
                    'page footer': QColor(0, 139, 139, 64),  # darkCyan with alpha
                    'list item': QColor(139, 0, 139, 64)   # magenta with alpha
                }
                
                # Get current page dimensions from MediaBox
                page = self.doc[self.current_page]
                mediabox = page.mediabox
                page_width = mediabox.width
                page_height = mediabox.height
                
                # Calculate scale factors based on displayed size
                display_scale = self.zoom
                
                for box in self.bounding_boxes:
                    try:
                        # Get relative coordinates and convert to absolute page coordinates
                        rel_x1 = box['coordinates'][0]['x']
                        rel_y1 = box['coordinates'][0]['y']
                        rel_x2 = box['coordinates'][2]['x']
                        rel_y2 = box['coordinates'][2]['y']
                        
                        # Convert to screen coordinates
                        x1 = int(rel_x1 * self.pixmap.width())
                        y1 = int(rel_y1 * self.pixmap.height())
                        x2 = int(rel_x2 * self.pixmap.width())
                        y2 = int(rel_y2 * self.pixmap.height())
                        
                        # Adjust for pan offset
                        x1 += self.pan_offset.x()
                        y1 += self.pan_offset.y()
                        x2 += self.pan_offset.x()
                        y2 += self.pan_offset.y()
                        
                        # Get category and corresponding color
                        category = box.get('category', 'text').lower()
                        color = category_colors.get(category, QColor(255, 0, 0, 64))  # red with alpha as default
                        
                        # Draw the rectangle with 2-pixel width and semi-transparent fill
                        pen = QPen(color, 2)
                        painter.setPen(pen)
                        color.setAlpha(64)
                        painter.setBrush(color)
                        painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                        
                        # Draw category label with semi-transparent background
                        label_color = color
                        label_color.setAlpha(128)  # Slightly more opaque for better text readability
                        painter.setPen(Qt.GlobalColor.black)
                        painter.setBrush(label_color)
                        # Draw a small rectangle for the label background
                        label_rect = painter.boundingRect(x1, y1 - 20, 100, 20, 
                                                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                                                        category)
                        painter.drawRect(label_rect)
                        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, category)
                        
                    except Exception as e:
                        logger.error(f"Error drawing bounding box: {str(e)}")
            painter.end()
            
    def display_page(self):
        if not self.doc:
            return
            
        try:
            page = self.doc[self.current_page]
            # Use a higher zoom factor for better quality
            display_zoom = self.zoom
            render_zoom = display_zoom * 2  # Double the zoom for rendering
            matrix = fitz.Matrix(render_zoom, render_zoom)
            pix = page.get_pixmap(matrix=matrix)
            
            # Convert to QImage
            img = QImage(pix.samples, pix.width, pix.height, 
                        pix.stride, QImage.Format.Format_RGB888)
            
            # Create QPixmap
            self.pixmap = QPixmap.fromImage(img)
            
            # Scale the pixmap to the display size
            display_size = self.size() * display_zoom
            self.pixmap = self.pixmap.scaled(display_size, 
                                           Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation)
            
            self.update()
            logger.debug(f"Displayed page {self.current_page + 1} at zoom {display_zoom:.2f} (render zoom: {render_zoom:.2f})")
        except Exception as e:
            logger.error(f"Error displaying page {self.current_page + 1}: {str(e)}")
            raise
        
    def set_zoom(self, new_zoom):
        old_zoom = self.zoom
        self.zoom = new_zoom
        
        # Calculate new pan offset to zoom around mouse position
        if hasattr(self, 'last_mouse_pos'):
            mouse_pos = self.last_mouse_pos
            old_pos = mouse_pos - self.pan_offset
            scale_factor = new_zoom / old_zoom
            new_pos = old_pos * scale_factor
            self.pan_offset = mouse_pos - new_pos
            
        self.display_page()
        logger.debug(f"Zoom changed from {old_zoom:.2f} to {new_zoom:.2f}")
        
    def wheelEvent(self, event):
        # Zoom in/out with mouse wheel
        delta = event.angleDelta().y()
        if delta > 0:
            self.set_zoom(self.zoom * 1.1)
        else:
            self.set_zoom(self.zoom / 1.1)
            
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_pan_pos = event.pos()
            
    def mouseMoveEvent(self, event):
        if self.last_pan_pos and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.pos() - self.last_pan_pos
            self.pan_offset += delta
            self.last_pan_pos = event.pos()
            self.update()
        self.last_mouse_pos = event.pos()
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_pan_pos = None

class ElementInfoDialog(QDialog):
    def __init__(self, element_data, parent=None):
        super().__init__(parent)
        self.element_data = element_data
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Element Information")
        layout = QVBoxLayout()
        
        # Create form layout for basic information
        form_layout = QFormLayout()
        form_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
        
        # Add type and page information
        form_layout.addRow("Type:", QLabel(self.element_data.get('type', '').capitalize()))
        form_layout.addRow("Page:", QLabel(str(self.element_data.get('page', ''))))
        
        layout.addLayout(form_layout)
        
        # Add caption in a read-only text area
        caption_label = QLabel("Caption:")
        caption_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout.addWidget(caption_label)
        
        caption_text = QTextEdit()
        caption_text.setReadOnly(True)
        caption_text.setPlainText(self.element_data.get('caption', ''))
        caption_text.setMaximumHeight(100)  # Limit height for captions
        caption_text.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout.addWidget(caption_text)
        
        # Add image if available
        pixmap = self.element_data.get('pixmap')
        if pixmap:
            # Create image label that will scale with the dialog
            image_label = QLabel()
            image_label.setPixmap(pixmap)
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            image_label.setMinimumSize(200, 200)  # Set minimum size for the image
            layout.addWidget(image_label, 1)  # Add stretch factor of 1
        
        # Add close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        button_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        
        # Set minimum size for the dialog
        self.setMinimumSize(400, 300)

class StructuredContentView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.current_doc = None  # Store current PDF document
        logger.debug("StructuredContentView initialized")
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Create toolbar for view mode switching
        toolbar = QToolBar()
        toolbar.setMovable(False)
        
        # Create view mode actions
        self.list_view_action = QAction("List View", self)
        self.list_view_action.setCheckable(True)
        self.list_view_action.setChecked(False)  # Changed to False
        self.list_view_action.triggered.connect(lambda: self.switch_view_mode('list'))
        
        self.icon_view_action = QAction("Icon View", self)
        self.icon_view_action.setCheckable(True)
        self.icon_view_action.setChecked(True)  # Changed to True
        self.icon_view_action.triggered.connect(lambda: self.switch_view_mode('icon'))
        
        # Add actions to toolbar
        toolbar.addAction(self.list_view_action)
        toolbar.addAction(self.icon_view_action)
        
        layout.addWidget(toolbar)
        
        # Create stacked widget to hold both views
        self.view_stack = QStackedWidget()
        
        # Create tree widget for list view
        self.content_tree = QTreeWidget()
        self.content_tree.setHeaderLabels(["Type", "Content", "Caption"])
        self.content_tree.setColumnWidth(0, 100)  # Type column
        self.content_tree.setColumnWidth(1, 300)  # Content column
        self.content_tree.setColumnWidth(2, 300)  # Caption column
        
        # Create grid widget for icon view
        self.content_grid = QGridLayout()
        self.grid_widget = QWidget()
        self.grid_widget.setLayout(self.content_grid)
        
        # Add views to stacked widget
        self.view_stack.addWidget(self.content_tree)
        self.view_stack.addWidget(self.grid_widget)
        
        # Set initial view to icon view
        self.view_stack.setCurrentIndex(1)  # Changed to 1 for icon view
        
        layout.addWidget(self.view_stack)
        self.setLayout(layout)
        
    def set_document(self, doc):
        """Set the current PDF document"""
        self.current_doc = doc
        
    def switch_view_mode(self, mode):
        """Switch between list and icon views"""
        if mode == 'list':
            self.view_stack.setCurrentIndex(0)
            self.list_view_action.setChecked(True)
            self.icon_view_action.setChecked(False)
        else:
            self.view_stack.setCurrentIndex(1)
            self.list_view_action.setChecked(False)
            self.icon_view_action.setChecked(True)
        
        # Update content in the current view
        if hasattr(self, '_current_content'):
            self.update_content(self._current_content)
        
    def _find_nearest_caption(self, element, page_elements):
        """Find the nearest caption for an element on the same page"""
        element_type = element.get('category', '').lower()
        element_coords = element.get('coordinates', [])
        
        if not element_coords or len(element_coords) < 4:
            return None
            
        # Get element's bounding box
        element_x1 = element_coords[0]['x']
        element_y1 = element_coords[0]['y']
        element_x2 = element_coords[2]['x']
        element_y2 = element_coords[2]['y']
        element_center_x = (element_x1 + element_x2) / 2
        element_center_y = (element_y1 + element_y2) / 2
        
        # Find all captions on the page
        captions = []
        for other_element in page_elements:
            if other_element.get('category', '').lower() == 'caption':
                caption_coords = other_element.get('coordinates', [])
                if caption_coords and len(caption_coords) >= 4:
                    caption_x1 = caption_coords[0]['x']
                    caption_y1 = caption_coords[0]['y']
                    caption_x2 = caption_coords[2]['x']
                    caption_y2 = caption_coords[2]['y']
                    caption_center_x = (caption_x1 + caption_x2) / 2
                    caption_center_y = (caption_y1 + caption_y2) / 2
                    
                    # Calculate distance between element and caption
                    dx = caption_center_x - element_center_x
                    dy = caption_center_y - element_center_y
                    distance = (dx * dx + dy * dy) ** 0.5
                    
                    captions.append({
                        'element': other_element,
                        'distance': distance,
                        'is_below': caption_y1 > element_y2,  # Caption is below the element
                        'is_above': caption_y2 < element_y1,  # Caption is above the element
                        'is_left': caption_x2 < element_x1,   # Caption is to the left
                        'is_right': caption_x1 > element_x2   # Caption is to the right
                    })
        
        if not captions:
            return None
            
        # Sort captions by distance
        captions.sort(key=lambda x: x['distance'])
        
        # For tables and figures, prefer captions below
        if element_type in ['table', 'figure']:
            below_captions = [c for c in captions if c['is_below']]
            if below_captions:
                return below_captions[0]['element']
        
        # For images and pictures, prefer captions below or to the right
        elif element_type in ['image', 'picture']:
            below_or_right = [c for c in captions if c['is_below'] or c['is_right']]
            if below_or_right:
                return below_or_right[0]['element']
        
        # If no preferred position found, return the closest caption
        return captions[0]['element']

    def _sort_elements_by_position(self, items):
        """Sort elements based on page number and position in a two-column layout"""
        def get_position_key(item):
            # First sort by page number
            page = item.get('page', 0)
            
            coords = item.get('coordinates', [])
            if not coords or len(coords) < 4:
                return (page, 0, 0)  # Default position if coordinates are missing
                
            # Get center point of the element
            x1 = coords[0]['x']
            y1 = coords[0]['y']
            x2 = coords[2]['x']
            y2 = coords[2]['y']
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Determine column (left or right)
            # Assuming page is divided into two equal columns
            is_left_column = center_x < 0.5
            
            # Determine vertical position (top or bottom)
            # Using 0.5 as the midpoint
            is_top = center_y < 0.5
            
            # Return a tuple for sorting:
            # First element: page number
            # Second element: 0 for left column, 1 for right column
            # Third element: 0 for top, 1 for bottom
            return (page, 0 if is_left_column else 1, 0 if is_top else 1)
        
        # Sort items based on their position
        return sorted(items, key=get_position_key)

    def _get_category_from_caption(self, caption):
        """Extract category from caption text (e.g., 'Figure 1. ...' -> 'figure')"""
        if not caption:
            return None
            
        # Split by whitespace and get first word
        first_word = caption.split()[0].lower()
        
        # Map common category words
        category_map = {
            'figure': 'figure',
            'fig': 'figure',
            'table': 'table',
            'image': 'image',
            'picture': 'picture',
            'photo': 'picture',
            'diagram': 'figure',
            'chart': 'figure',
            'graph': 'figure'
        }
        
        return category_map.get(first_word)

    def update_content(self, page_structures):
        """Update the structured content view with page structures"""
        self._current_content = page_structures  # Store for view switching
        
        # Clear current view
        self.content_tree.clear()
        self._clear_grid()
        
        # Group content by type
        content_by_type = {
            'image': [],
            'table': [],
            'figure': [],
            'picture': []
        }
        
        # Process each page's structure
        for page_num, structure in page_structures.items():
            elements = structure.get('structure', {}).get('elements', [])
            for element in elements:
                element_type = element.get('category', '').lower()
                if element_type in content_by_type:
                    # Find nearest caption
                    caption_element = self._find_nearest_caption(element, elements)
                    caption = caption_element.get('content', {}).get('text', '') if caption_element else None
                    
                    # Try to get category from caption
                    if caption:
                        caption_category = self._get_category_from_caption(caption)
                        if caption_category:
                            element_type = caption_category
                    
                    content_by_type[element_type].append({
                        'page': int(page_num) + 1,  # Convert to 1-based page number
                        'content': element.get('content', {}).get('text', ''),
                        'caption': caption,
                        'coordinates': element.get('coordinates', []),
                        'page_width': element.get('attributes', {}).get('page_width', 0),
                        'page_height': element.get('attributes', {}).get('page_height', 0)
                    })
        
        # Sort elements by position for each type
        for content_type in content_by_type:
            content_by_type[content_type] = self._sort_elements_by_position(content_by_type[content_type])
        
        # Update current view
        if self.view_stack.currentIndex() == 0:
            self._update_list_view(content_by_type)
        else:
            self._update_icon_view(content_by_type)
    
    def _update_list_view(self, content_by_type):
        """Update the list view with content"""
        for content_type, items in content_by_type.items():
            if items:
                type_item = QTreeWidgetItem(self.content_tree)
                type_item.setText(0, content_type.capitalize())
                type_item.setExpanded(True)
                
                for item in items:
                    content_item = QTreeWidgetItem(type_item)
                    content_item.setText(0, f"Page {item['page']}")
                    content_item.setText(1, item['content'])
                    content_item.setText(2, item['caption'] if item['caption'] else '')
    
    def _update_icon_view(self, content_by_type):
        """Update the icon view with content"""
        self._clear_grid()
        
        if not self.current_doc:
            logger.warning("No PDF document available for icon view")
            return
            
        row = 0
        col = 0
        max_cols = 3  # Number of items per row
        
        for content_type, items in content_by_type.items():
            if items:
                # Add type label (single line)
                type_label = QLabel(content_type.capitalize())
                type_label.setStyleSheet("font-weight: bold;")
                self.content_grid.addWidget(type_label, row, 0, 1, max_cols)
                row += 1
                
                # Add items
                for item in items:
                    # Create item widget
                    item_widget = QWidget()
                    item_layout = QVBoxLayout()
                    item_layout.setContentsMargins(5, 5, 5, 5)
                    
                    # Get page pixmap
                    try:
                        page = self.current_doc[item['page'] - 1]  # Convert to 0-based index
                        zoom = 4  # Higher zoom for better quality (4x)
                        matrix = fitz.Matrix(zoom, zoom)
                        pix = page.get_pixmap(matrix=matrix)
                        
                        # Convert to QImage
                        img = QImage(pix.samples, pix.width, pix.height, 
                                   pix.stride, QImage.Format.Format_RGB888)
                        
                        # Get coordinates
                        coords = item['coordinates']
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
                            max_size = 400  # Increased maximum size for the image
                            pixmap = pixmap.scaled(max_size, max_size, 
                                                 Qt.AspectRatioMode.KeepAspectRatio,
                                                 Qt.TransformationMode.SmoothTransformation)
                            
                            # Create image label
                            image_label = QLabel()
                            image_label.setPixmap(pixmap)
                            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                            item_layout.addWidget(image_label)
                            
                            # Store pixmap for dialog
                            item_pixmap = pixmap
                        else:
                            item_pixmap = None
                    except Exception as e:
                        logger.error(f"Error creating image for {content_type} on page {item['page']}: {str(e)}")
                        item_pixmap = None
                    
                    # Show caption for tables and figures, content for others
                    if content_type in ['table', 'figure']:
                        if item['caption']:
                            caption_label = QLabel(item['caption'])
                            caption_label.setWordWrap(True)
                            caption_label.setStyleSheet("font-style: italic;")
                            item_layout.addWidget(caption_label)
                    else:
                        content_label = QLabel(item['content'])
                        content_label.setWordWrap(True)
                        item_layout.addWidget(content_label)
                    
                    item_widget.setLayout(item_layout)
                    item_widget.setStyleSheet("border: 1px solid #ccc; padding: 5px;")
                    
                    # Store element data for double-click
                    item_widget.element_data = {
                        'type': content_type,
                        'page': item['page'],
                        'caption': item['caption'],
                        'content': item['content'],
                        'pixmap': item_pixmap
                    }
                    
                    # Add double-click handler
                    item_widget.mouseDoubleClickEvent = lambda event, widget=item_widget: self._show_element_info(widget)
                    
                    # Add to grid
                    self.content_grid.addWidget(item_widget, row, col)
                    col += 1
                    if col >= max_cols:
                        col = 0
                        row += 1
                
                # Add spacing after each type
                row += 1
    
    def _clear_grid(self):
        """Clear the grid layout"""
        while self.content_grid.count():
            item = self.content_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def _is_caption_for_element(self, caption_element, target_element):
        """Check if a caption element belongs to a target element"""
        # Get coordinates
        caption_coords = caption_element.get('coordinates', [])
        target_coords = target_element.get('coordinates', [])
        
        if not caption_coords or not target_coords:
            return False
            
        # Get y-coordinates (vertical position)
        caption_y = caption_coords[0]['y']
        target_y = target_coords[0]['y']
        
        # Check if caption is below the target element
        # and within a reasonable distance (e.g., 0.1 of page height)
        return caption_y > target_y and (caption_y - target_y) < 0.1

    def _show_element_info(self, widget):
        """Show detailed information about an element"""
        if hasattr(widget, 'element_data'):
            dialog = ElementInfoDialog(widget.element_data, self)
            dialog.exec()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{PROGRAM_NAME} v{PROGRAM_VERSION}")
        
        # Initialize database
        self.init_database()
        
        # Load window geometry from settings
        self.settings = QSettings(COMPANY_NAME, PROGRAM_NAME)
        geometry = self.settings.value("window_geometry")
        if geometry:
            self.restoreGeometry(geometry)
        else:
            # Default geometry if no settings exist
            self.setGeometry(100, 100, 1200, 800)
        
        # Set window icon
        icon = setup_icons()
        if icon:
            self.setWindowIcon(icon)
            logger.info("Set application icon")

        self.document_data = {
            'page_structures': {},
            'metadata': {}
        }
        self.recent_files = []

        # Create status bar
        self.status_label = self.statusBar()
        self.status_label.showMessage("Ready")
        
        # Create directory tree widget
        self.create_directory_tree()
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        
        # Create PDF viewer tab
        self.pdf_viewer = PDFViewer()
        self.tab_widget.addTab(self.pdf_viewer, "PDF View")
        
        # Create structured content tab
        self.structured_view = StructuredContentView()
        self.tab_widget.addTab(self.structured_view, "Structured Content")
        
        self.create_toolbar()
        self.create_menu()
        logger.info(f"{PROGRAM_NAME} v{PROGRAM_VERSION} started")

        # Add recent files list
        self.load_recent_files()

    def init_database(self):
        """Initialize the database"""
        try:
            # Initialize database with migrations
            init_database(DB_PATH)
            
        except Exception as e:
            QMessageBox.critical(self, "Database Error", 
                f"Error initializing database: {str(e)}")
            logger.error(f"Database initialization error: {str(e)}")
            self.close()

    def create_directory_tree(self):
        """Create and setup the directory tree widgets"""
        # Create main dock widget
        self.library_dock = QDockWidget("PDF Library", self)
        self.library_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | 
                                        Qt.DockWidgetArea.RightDockWidgetArea)
        
        # Create a widget to hold the splitter
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Create collections tree widget
        self.collections_tree = QTreeWidget()
        self.collections_tree.setHeaderLabels(["Collections"])
        self.collections_tree.setColumnWidth(0, 200)
        self.collections_tree.setHeaderHidden(True)  # Hide header
        self.collections_tree.itemClicked.connect(self.collection_clicked)
        
        # Create items tree widget
        self.items_tree = QTreeWidget()
        self.items_tree.setHeaderLabels(["Items"])
        self.items_tree.setColumnWidth(0, 300)
        self.items_tree.setHeaderHidden(True)  # Hide header
        self.items_tree.itemClicked.connect(self.item_clicked)
        
        # Add widgets to splitter
        splitter.addWidget(self.collections_tree)
        splitter.addWidget(self.items_tree)
        
        # Set initial sizes (40% for collections, 60% for items)
        splitter.setSizes([400, 600])
        
        # Add splitter to layout
        layout.addWidget(splitter)
        
        # Set the container as the dock widget's widget
        self.library_dock.setWidget(container)
        
        # Add the dock widget to the main window
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.library_dock)
        
        logger.info("Directory tree widgets created with splitter")

    def collection_clicked(self, item, column):
        """Handle click on collection item"""
        if hasattr(item, 'collection_id'):
            # Clear the items tree
            self.items_tree.clear()
            
            # Clear the PDF viewer
            self.pdf_viewer.set_bounding_boxes([])
            self.pdf_viewer.pixmap = None
            self.pdf_viewer.update()
            
            # Clear current document data
            self.document_data = {
                'page_structures': {},
                'metadata': {}
            }
            
            # Get all items for this collection
            if hasattr(self, 'collection_data') and item.collection_id in self.collection_data:
                for item_data in self.collection_data[item.collection_id]:
                    # Create a new tree item
                    new_item = QTreeWidgetItem()
                    new_item.setText(0, item_data['text'])
                    
                    # Copy relevant attributes
                    if 'item_id' in item_data:
                        new_item.item_id = item_data['item_id']
                    if 'file_path' in item_data:
                        new_item.file_path = item_data['file_path']
                    
                    # If item data has children (PDF attachments), copy them
                    for child_data in item_data.get('children', []):
                        child_item = QTreeWidgetItem()
                        child_item.setText(0, child_data['text'])
                        if 'file_path' in child_data:
                            child_item.file_path = child_data['file_path']
                        new_item.addChild(child_item)
                    
                    self.items_tree.addTopLevelItem(new_item)
            
            logger.debug(f"Showing items for collection: {item.text(0)}")

    def item_clicked(self, item, column):
        """Handle click on item in items tree"""
        if hasattr(item, 'file_path') and item.file_path:
            # Direct click on a PDF item
            if item.file_path.lower().endswith('.pdf'):
                self.load_pdf_file(item.file_path)
                logger.debug(f"Loading PDF file: {item.file_path}")
        else:
            # Click on a parent item - check for PDF attachments
            pdf_items = []
            for i in range(item.childCount()):
                child = item.child(i)
                if (hasattr(child, 'file_path') and 
                    child.file_path and 
                    child.file_path.lower().endswith('.pdf')):
                    pdf_items.append(child)
            
            if len(pdf_items) == 1:
                # If there's exactly one PDF, load it
                pdf_item = pdf_items[0]
                self.load_pdf_file(pdf_item.file_path)
                logger.debug(f"Loading single PDF attachment: {pdf_item.file_path}")
            elif len(pdf_items) > 1:
                # If there are multiple PDFs, load the first one and log a message
                pdf_item = pdf_items[0]
                self.load_pdf_file(pdf_item.file_path)
                logger.info(f"Loading first of {len(pdf_items)} PDF attachments: {pdf_item.file_path}")

    def load_pdf_file(self, file_path):
        """Load a PDF file and try to load its analysis from database"""
        self.current_file = file_path
        self.pdf_directory = os.path.dirname(file_path)
        self.document_data = {
            'page_structures': {},
            'metadata': {}
        }
        
        # Try to find the item in the items tree to get Zotero key
        zotero_key = None
        current_item = None
        for i in range(self.items_tree.topLevelItemCount()):
            item = self.items_tree.topLevelItem(i)
            if hasattr(item, 'file_path') and item.file_path == file_path:
                current_item = item
                break
            # Check child items if no match found
            for j in range(item.childCount()):
                child = item.child(j)
                if hasattr(child, 'file_path') and child.file_path == file_path:
                    current_item = child
                    break
            if current_item:
                break
        
        # Get Zotero key from the item if available
        if current_item and hasattr(current_item, 'zotero_key'):
            zotero_key = current_item.zotero_key
            logger.debug(f"Found Zotero key for file: {zotero_key}")
        
        # If no Zotero key found in tree, try to extract from path
        if not zotero_key and 'storage' in file_path:
            storage_dir = os.path.dirname(file_path)
            potential_key = os.path.basename(storage_dir)
            if len(potential_key) == 8:  # Zotero keys are 8 characters
                zotero_key = potential_key
                logger.debug(f"Extracted Zotero key from path: {zotero_key}")
        
        self.doc = fitz.open(file_path)
        self.current_page = 0
        self.pdf_viewer.current_page = 0
        self.pdf_viewer.open_pdf(file_path)
        self.update_navigation()
        self.status_label.showMessage(f"Opened: {os.path.basename(file_path)}", 3000)
        logger.info(f"Opened PDF file: {self.current_file}")
        
        # Set document in structured view
        self.structured_view.set_document(self.doc)
        
        # Try to load analysis from database
        if self.load_analysis_from_database(file_path, zotero_key):
            self.update_page_display()
            # Update structured content view with all page structures
            self.structured_view.update_content(self.document_data['page_structures'])
            self.status_label.showMessage(f"Loaded analysis for {os.path.basename(file_path)}", 3000)
        else:
            # Try to load session from database
            try:
                self.load_session(file_path)
                logger.info(f"Loaded session data for {file_path}")
                # Update structured content view with all page structures
                self.structured_view.update_content(self.document_data['page_structures'])
                self.status_label.showMessage(f"Loaded session data for {os.path.basename(file_path)}", 3000)
            except Exception as e:
                logger.error(f"Error loading session data: {str(e)}")

    def load_directory(self):
        """Load all PDF files from a selected directory and its subdirectories"""
        try:
            # Set wait cursor
            QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
            self.status_label.showMessage("Loading directory...", 0)
            QApplication.processEvents()
            
            dir_path = QFileDialog.getExistingDirectory(
                self, "Select Directory", "",
                QFileDialog.Option.ShowDirsOnly
            )
            
            if not dir_path:
                return
                
            # Clear existing items
            self.collections_tree.clear()
            self.items_tree.clear()
            
            # Create root collection item
            root_item = QTreeWidgetItem()
            root_item.setText(0, os.path.basename(dir_path))
            root_item.dir_path = dir_path
            self.collections_tree.addTopLevelItem(root_item)
            
            # Dictionary to store directory items by path
            self.directory_items = {dir_path: root_item}
            pdf_files = []
            
            def load_subdirectories(parent_item, current_dir):
                try:
                    # Get all items in the directory
                    items = os.listdir(current_dir)
                    items.sort()  # Sort alphabetically
                    
                    # First process directories
                    for item in sorted(items):
                        full_path = os.path.join(current_dir, item)
                        if os.path.isdir(full_path):
                            # Create directory item
                            dir_item = QTreeWidgetItem(parent_item)
                            dir_item.setText(0, item)
                            dir_item.dir_path = full_path
                            self.directory_items[full_path] = dir_item
                            
                            # Recursively load subdirectories
                            load_subdirectories(dir_item, full_path)
                    
                    # Then collect PDF files
                    for item in sorted(items):
                        full_path = os.path.join(current_dir, item)
                        if item.lower().endswith('.pdf'):
                            pdf_files.append(full_path)
                            
                except Exception as e:
                    logger.error(f"Error loading subdirectory {current_dir}: {str(e)}")
            
            # Start recursive loading
            load_subdirectories(root_item, dir_path)
            
            if not pdf_files:
                QMessageBox.information(self, "No PDF Files", 
                                    "No PDF files found in the selected directory or its subdirectories.")
                return
                
            # Sort files by name
            pdf_files.sort()
            
            # Add PDF files to items tree
            for pdf_file in pdf_files:
                item = QTreeWidgetItem()
                item.setText(0, os.path.basename(pdf_file))
                item.file_path = pdf_file
                self.items_tree.addTopLevelItem(item)
            
            # Load the first PDF file
            if pdf_files:
                self.load_pdf_file(pdf_files[0])
                
                # Store the list of PDF files
                self.pdf_files = pdf_files
                self.current_file_index = 0
                
            # Expand the first level of directories
            for i in range(self.collections_tree.topLevelItemCount()):
                self.collections_tree.topLevelItem(i).setExpanded(True)
                
            self.status_label.showMessage(f"Loaded directory: {dir_path}", 3000)
            logger.info(f"Loaded directory: {dir_path} with {len(pdf_files)} PDF files")
            
        except Exception as e:
            logger.error(f"Error loading directory: {str(e)}")
            QMessageBox.warning(self, "Error", f"Could not load directory:\n{str(e)}")
            
        finally:
            # Restore cursor
            QApplication.restoreOverrideCursor()
            self.ensure_normal_cursor()

    def create_menu(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open PDF", self)
        open_action.triggered.connect(self.open_pdf)
        file_menu.addAction(open_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        
        preferences_action = QAction("Preferences", self)
        preferences_action.triggered.connect(self.show_preferences)
        edit_menu.addAction(preferences_action)
        
    def create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        # Open file action
        open_action = QAction("Open PDF", self)
        open_action.triggered.connect(self.open_pdf)
        toolbar.addAction(open_action)
        
        # Load directory action
        load_dir_action = QAction("Load Directory", self)
        load_dir_action.triggered.connect(self.load_directory)
        toolbar.addAction(load_dir_action)
        
        # Load Zotero library action
        load_zotero_action = QAction("Load Zotero Library", self)
        load_zotero_action.triggered.connect(self.load_zotero_library)
        toolbar.addAction(load_zotero_action)
        
        # Add separator
        toolbar.addSeparator()
        
        # Navigation buttons
        self.prev_button = QAction("Previous", self)
        self.prev_button.triggered.connect(self.prev_page)
        self.prev_button.setEnabled(False)
        toolbar.addAction(self.prev_button)
        
        # Current page input
        self.current_page_input = QLineEdit()
        self.current_page_input.setFixedWidth(50)
        self.current_page_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_page_input.setText("0")
        self.current_page_input.returnPressed.connect(self.go_to_page)
        toolbar.addWidget(self.current_page_input)
        
        # Total pages label
        self.total_pages_label = QLabel("/ 0")
        toolbar.addWidget(self.total_pages_label)
        
        # Next button
        self.next_button = QAction("Next", self)
        self.next_button.triggered.connect(self.next_page)
        self.next_button.setEnabled(False)
        toolbar.addAction(self.next_button)
        
        # Add separator
        toolbar.addSeparator()
        
        # Analyze action
        analyze_action = QAction("Analyze", self)
        analyze_action.triggered.connect(lambda: self.analyze_pdf(self.current_file) if self.current_file else None)
        toolbar.addAction(analyze_action)

        # Add separator
        toolbar.addSeparator()

        # Batch analyze action
        batch_analyze_action = QAction("Batch Analyze", self)
        batch_analyze_action.triggered.connect(self.batch_analyze)
        toolbar.addAction(batch_analyze_action)

    def load_recent_files(self):
        """Load recent files from settings"""
        #settings = QSettings()
        files = self.settings.value('recent_files', [])
        # Convert to list if it's a string (can happen with QSettings)
        if isinstance(files, str):
            files = [files]
        elif not isinstance(files, list):
            files = []
        
        # Filter out non-existent files
        self.recent_files = [f for f in files if os.path.exists(f)]
        # Keep only last 5 files
        self.recent_files = self.recent_files[:5]
        logger.debug(f"Loaded recent files: {self.recent_files}")
        
        if hasattr(self, 'recent_files_menu'):
            self.update_recent_files_menu()

    def add_to_recent_files(self, file_path):
        """Add a file to recent files list"""
        if file_path in self.recent_files:
            logger.debug(f"Removing {file_path} from recent files")
            self.recent_files.remove(file_path)
        logger.debug(f"Adding {file_path} to recent files")
        self.recent_files.insert(0, file_path)
        # Keep only last 5 files
        self.recent_files = self.recent_files[:5]
        self.save_recent_files()
        self.update_recent_files_menu()
        
    def update_recent_files_menu(self):
        """Update the recent files menu"""
        self.recent_files_menu.clear()
        
        if not self.recent_files:
            no_recent = self.recent_files_menu.addAction("No recent files")
            no_recent.setEnabled(False)
            return
            
        for file_path in self.recent_files:
            action = self.recent_files_menu.addAction(os.path.basename(file_path))
            action.setData(file_path)
            action.setStatusTip(file_path)
            action.triggered.connect(self.open_recent_file)  # Connect the triggered signal
        
    def open_recent_file(self):
        """Handle opening a recent file"""
        logger.info("Opening recent file")
        action = self.sender()
        if action:
            file_path = action.data()
            logger.info(f"Opening recent file: {file_path}")
            if os.path.exists(file_path):
                self.load_pdf(file_path)
            else:
                QMessageBox.warning(self, "File Not Found", 
                                  f"The file {file_path} no longer exists.")
                self.recent_files.remove(file_path)
                self.save_recent_files()
                self.update_recent_files_menu()

    def show_preferences(self):
        dialog = PreferencesDialog(self)
        dialog.exec()
        
    def open_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open PDF File", "", "PDF Files (*.pdf)"
        )
        if file_path:
            self.current_file = file_path
            self.pdf_directory = os.path.dirname(file_path)
            self.document_data = {
                'page_structures': {},
                'metadata': {}
            }
            self.doc = fitz.open(file_path)
            self.current_page = 0
            self.pdf_viewer.current_page = 0
            self.pdf_viewer.open_pdf(file_path)
            self.update_navigation()
            self.status_label.showMessage(f"Opened: {os.path.basename(file_path)}", 3000)
            logger.info(f"Opened PDF file: {self.current_file}")
            
            # Try to load analysis from database
            if self.load_analysis_from_database(file_path):
                self.update_page_display()
                self.status_label.showMessage(f"Loaded analysis for {os.path.basename(file_path)}", 3000)
            else:
                # Try to load session from database
                try:
                    self.load_session(file_path)
                    logger.info(f"Loaded session data for {file_path}")
                    self.status_label.showMessage(f"Loaded session data for {os.path.basename(file_path)}", 3000)
                except Exception as e:
                    logger.error(f"Error loading session data: {str(e)}")

    def ensure_normal_cursor(self):
        """Make sure the cursor is restored to normal"""
        # Restore cursor state if it's been overridden
        while QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()

    def analyze_pdf(self, file_path):
        """Analyze the PDF file and store results in the database."""
        try:
            # Calculate file hash
            file_hash = calculate_file_hash(file_path)
            
            # Check if analysis exists in database
            try:
                document = PDFDocument.get(PDFDocument.file_hash == file_hash)
                logger.info(f"Found existing analysis for {file_path}")
                
                # Load existing analysis
                analyses = (PageAnalysis
                          .select()
                          .where(PageAnalysis.document == document)
                          .order_by(PageAnalysis.page_number))
                
                if analyses.count() > 0:
                    # Load session data if exists
                    session = (SessionData
                             .select()
                             .where(SessionData.document == document)
                             .order_by(SessionData.last_accessed.desc())
                             .first())
                    
                    if session:
                        # Load session data
                        session_data = json.loads(session.session_data)
                        self.document_data = session_data.get('document_data', {})
                        self.current_page = session.current_page
                        self.pdf_viewer.current_page = self.current_page
                        
                        # Update UI
                        self.update_page_display()
                        self.status_label.showMessage(f"Loaded existing analysis for {os.path.basename(file_path)}", 3000)
                        logger.info(f"Loaded existing analysis and session data for {file_path}")
                        return True
                    
            except DoesNotExist:
                logger.info(f"No existing analysis found for {file_path}")
                document = None
            
            # Perform new analysis
            logger.info(f"Analyzing PDF: {file_path}")
            self.status_label.showMessage("Analyzing PDF...", 0)  # 0 means show until changed
            QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
            
            try:
                # Get service URL from settings
                settings = QSettings(COMPANY_NAME, PROGRAM_NAME)
                base_url = settings.value("service/url", "").rstrip('/')
                if not base_url:
                    raise ValueError("Service URL not configured")
                
                logger.info(f"Using analysis service at: {base_url}")
                
                # First try PyMuPDF text extraction
                sample_size = max(1, len(self.doc) // 10)  # 10% of pages or at least 1 page
                logger.info(f"Checking first {sample_size} pages for text content using PyMuPDF")
                
                # Check text content in sample pages using PyMuPDF
                page_char_counts = {}
                for page_num in range(sample_size):
                    try:
                        page = self.doc[page_num]
                        text = page.get_text()
                        char_count = len(text.strip())
                        page_char_counts[page_num] = char_count
                        logger.debug(f"Page {page_num + 1}: {char_count} characters")
                    except Exception as e:
                        logger.error(f"Error extracting text from page {page_num + 1}: {str(e)}")
                        page_char_counts[page_num] = 0
                
                # Calculate average characters per page
                total_chars = sum(page_char_counts.values())
                avg_chars_per_page = total_chars / len(page_char_counts) if page_char_counts else 0
                
                logger.info(f"PyMuPDF found average of {avg_chars_per_page:.1f} characters per page in sample")
                logger.info(f"Character counts by page: {page_char_counts}")
                
                # Decide if OCR is needed:
                # - If average characters per page is less than 100
                # - Or if more than half of sample pages have less than 50 characters
                low_text_pages = sum(1 for count in page_char_counts.values() if count < 50)
                need_ocr = (avg_chars_per_page < 100 or 
                           (low_text_pages / len(page_char_counts) > 0.5))
                
                if need_ocr:
                    logger.info(f"Low text content detected in PyMuPDF extraction (avg {avg_chars_per_page:.1f} chars/page), proceeding with OCR")
                    
                    # Perform OCR
                    url_ocr = f"{base_url}/ocr"
                    with open(file_path, 'rb') as pdf_file:
                        files = {
                            'file': (os.path.basename(file_path), pdf_file, 'application/pdf')
                        }
                        
                        self.status_label.showMessage("Performing OCR on document...", 0)
                        QApplication.processEvents()
                        
                        try:
                            response = requests.post(url_ocr, files=files)
                            
                            if response.status_code == 200:
                                ocr_pdf_path = os.path.join(os.path.dirname(file_path), 
                                                          'ocr_' + os.path.basename(file_path))
                                
                                with open(ocr_pdf_path, 'wb') as f:
                                    f.write(response.content)
                                
                                logger.info(f"OCR PDF saved to: {ocr_pdf_path}")
                                
                                # Use the OCR'd PDF for layout analysis
                                analysis_file = ocr_pdf_path
                            else:
                                raise ValueError(f"OCR service returned status code: {response.status_code}")
                        except requests.exceptions.ConnectionError:
                            error_msg = f"Could not connect to the OCR service at {url_ocr}. Please check the service URL in preferences."
                            logger.error(error_msg)
                            QMessageBox.warning(self, "Connection Error", error_msg)
                            return False
                        except Exception as e:
                            error_msg = f"Error during OCR: {str(e)}"
                            logger.error(error_msg)
                            QMessageBox.warning(self, "OCR Error", error_msg)
                            return False
                else:
                    logger.info("Sufficient text content found in PDF, proceeding with layout analysis")
                    analysis_file = file_path
                
                # Perform layout analysis
                with open(analysis_file, 'rb') as pdf_file:
                    files = {
                        'file': (os.path.basename(analysis_file), pdf_file, 'application/pdf')
                    }
                    
                    self.status_label.showMessage("Analyzing document structure...", 0)
                    QApplication.processEvents()
                    
                    try:
                        response = requests.post(base_url, files=files)
                        
                        if response.status_code == 200:
                            results = response.json()
                            
                            if not isinstance(results, list):
                                raise ValueError("Expected JSON array response")
                            
                            # Process analysis results
                            page_elements = {}
                            for element in results:
                                page_num = str(element['page_number'] - 1)  # Convert to 0-based index
                                if page_num not in page_elements:
                                    page_elements[page_num] = []
                                
                                # Convert the element to our structure format
                                structured_element = {
                                    'category': element['type'].lower(),
                                    'content': {
                                        'text': element['text'],
                                        'html': element['text']
                                    },
                                    'coordinates': [
                                        {'x': element['left'] / element['page_width'], 
                                         'y': element['top'] / element['page_height']},
                                        {'x': (element['left'] + element['width']) / element['page_width'],
                                         'y': element['top'] / element['page_height']},
                                        {'x': (element['left'] + element['width']) / element['page_width'],
                                         'y': (element['top'] + element['height']) / element['page_height']},
                                        {'x': element['left'] / element['page_width'],
                                         'y': (element['top'] + element['height']) / element['page_height']}
                                    ],
                                    'relative_size': {
                                        'width_mm': element['width'] * 0.352778,
                                        'height_mm': element['height'] * 0.352778,
                                        'point_size': element['height'] * 0.75
                                    },
                                    'attributes': {
                                        'page_width': element['page_width'],
                                        'page_height': element['page_height']
                                    },
                                    'id': len(page_elements[page_num])  # ID starts from 0 for each page
                                }
                                page_elements[page_num].append(structured_element)
                            
                            # Store the analysis results
                            for page_num, elements in page_elements.items():
                                self.document_data['page_structures'][page_num] = {
                                    'timestamp': datetime.datetime.now().isoformat(),
                                    'structure': {
                                        'elements': elements,
                                        'metadata': {
                                            'page_number': int(page_num),
                                            'page_width': elements[0]['attributes']['page_width'],
                                            'page_height': elements[0]['attributes']['page_height']
                                        }
                                    }
                                }
                            
                            # Update the display
                            self.update_page_display()
                            
                            # Auto-save session after analysis
                            self.save_session()
                            
                            # Show completion message
                            analyzed_pages = len(page_elements)
                            msg = f"Document analysis completed - {analyzed_pages} pages analyzed"
                            
                            self.status_label.showMessage(msg, 3000)
                            logger.info(msg)
                            return True
                            
                        else:
                            error_msg = f"Layout analysis service returned status code: {response.status_code}"
                            logger.error(error_msg)
                            QMessageBox.warning(self, "Analysis Error", error_msg)
                            return False
                            
                    except requests.exceptions.ConnectionError:
                        error_msg = f"Could not connect to the layout analysis service at {base_url}. Please check the service URL in preferences."
                        logger.error(error_msg)
                        QMessageBox.warning(self, "Connection Error", error_msg)
                        return False
                    except Exception as e:
                        error_msg = f"Error during layout analysis: {str(e)}"
                        logger.error(error_msg)
                        logger.error("Full error details:", exc_info=True)
                        QMessageBox.warning(self, "Analysis Error", error_msg)
                        return False
                        
            finally:
                # Restore cursor
                QApplication.restoreOverrideCursor()
                self.ensure_normal_cursor()
                
        except Exception as e:
            logger.error(f"Error in analyze_pdf: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error processing PDF: {str(e)}")
            return False

    def load_analysis_from_database(self, file_path, zotero_key=None):
        """Load analysis results from database for a given file"""
        try:
            with db:
                # Try to find document by Zotero key first
                document = None
                if zotero_key:
                    try:
                        document = PDFDocument.get(PDFDocument.zotero_key == zotero_key)
                        logger.debug(f"Found document by Zotero key: {zotero_key}")
                    except DoesNotExist:
                        logger.debug(f"No document found with Zotero key: {zotero_key}")
                
                # If no document found by Zotero key, try file path
                if not document:
                    try:
                        document = PDFDocument.get(PDFDocument.file_path == file_path)
                        logger.debug(f"Found document by file path: {file_path}")
                    except DoesNotExist:
                        logger.debug(f"No document found with file path: {file_path}")
                        return False
                
                if document.last_analyzed:
                    pages = (PageAnalysis
                           .select()
                           .where(PageAnalysis.document == document)
                           .order_by(PageAnalysis.page_number))
                    
                    for page in pages:
                        page_num = str(page.page_number)
                        self.document_data['page_structures'][page_num] = json.loads(page.analysis_data)
                    
                    logger.info(f"Loaded analysis from database for {file_path}")
                    return True
        except Exception as e:
            logger.error(f"Error loading analysis from database: {str(e)}")
        
        return False

    def save_session(self):
        """Save current session data to database"""
        try:
            if not hasattr(self, 'current_file') or not self.current_file:
                logger.debug("No current file to save session for")
                return
                
            with db:
                # Get or create PDFDocument using Zotero key if available
                file_hash = None
                zotero_key = None
                
                # Try to find the current item in the items tree
                current_item = None
                for i in range(self.items_tree.topLevelItemCount()):
                    item = self.items_tree.topLevelItem(i)
                    if hasattr(item, 'file_path') and item.file_path == self.current_file:
                        current_item = item
                        break
                    # Check child items if no match found
                    for j in range(item.childCount()):
                        child = item.child(j)
                        if hasattr(child, 'file_path') and child.file_path == self.current_file:
                            current_item = child
                            break
                    if current_item:
                        break
                
                # Get Zotero key from the item if available
                if current_item and hasattr(current_item, 'zotero_key'):
                    zotero_key = current_item.zotero_key
                    logger.debug(f"Found Zotero key for current file: {zotero_key}")
                
                # If no Zotero key found, use file hash
                if not zotero_key:
                    file_hash = calculate_file_hash(self.current_file)
                    logger.debug(f"No Zotero key found, using file hash: {file_hash}")
                
                # Try to find existing document
                document = None
                try:
                    # First try by file path
                    document = PDFDocument.get(PDFDocument.file_path == self.current_file)
                    logger.debug(f"Found existing document by file path: {self.current_file}")
                except DoesNotExist:
                    try:
                        # Then try by Zotero key if available
                        if zotero_key:
                            document = PDFDocument.get(PDFDocument.zotero_key == zotero_key)
                            logger.debug(f"Found existing document by Zotero key: {zotero_key}")
                    except DoesNotExist:
                        try:
                            # Finally try by file hash if available
                            if file_hash:
                                document = PDFDocument.get(PDFDocument.file_hash == file_hash)
                                logger.debug(f"Found existing document by file hash: {file_hash}")
                        except DoesNotExist:
                            # Create new document if none found
                            document = PDFDocument.create(
                                file_path=self.current_file,
                                file_hash=file_hash,
                                zotero_key=zotero_key,
                                title=os.path.basename(self.current_file),
                                page_count=len(self.doc) if self.doc else 0
                            )
                            logger.debug(f"Created new document record")
                
                # Update document fields if needed
                if document:
                    if not document.file_hash and file_hash:
                        document.file_hash = file_hash
                    if not document.zotero_key and zotero_key:
                        document.zotero_key = zotero_key
                    if not document.page_count and self.doc:
                        document.page_count = len(self.doc)
                    document.save()
                    logger.debug(f"Updated document record")
                
                # Prepare session data
                session_data = {
                    'document_data': {
                        'page_structures': self.document_data['page_structures'],
                        'metadata': self.document_data['metadata'],
                        'page_dimensions': self.document_data.get('page_dimensions', {})
                    },
                    'session_info': {
                        'analyzed_pages': len(set(self.document_data['page_structures'].keys())),
                        'app_version': PROGRAM_VERSION,
                        'zotero_key': zotero_key
                    }
                }
                
                # Create new session entry
                SessionData.create(
                    document=document,
                    current_page=self.current_page,
                    session_data=json.dumps(session_data, ensure_ascii=False),
                    last_accessed=datetime.datetime.now()
                )
                
                logger.info(f"Session saved to database for {self.current_file} (Zotero key: {zotero_key})")
                logger.debug(f"Saved {len(self.document_data['page_structures'])} page structures")
                
        except Exception as e:
            logger.error(f"Error saving session: {str(e)}")
            logger.error("Full error details:", exc_info=True)

    def load_session(self, file_path=None):
        """Load a previously saved session from database"""
        try:
            if not file_path:
                file_path, _ = QFileDialog.getOpenFileName(
                    self, "Load Session", "", "PDF Files (*.pdf)"
                )
                if not file_path:
                    return

            logger.info(f"Loading session for {file_path}")
            
            with db:
                # Try to find document by Zotero key first
                document = None
                zotero_key = None
                
                # Extract Zotero key from file path
                storage_dir = os.path.dirname(file_path)
                if 'storage' in storage_dir:
                    # The Zotero key is typically the directory name in the storage folder
                    potential_key = os.path.basename(storage_dir)
                    if len(potential_key) == 8:  # Zotero keys are 8 characters
                        zotero_key = potential_key
                        logger.debug(f"Extracted Zotero key from path: {zotero_key}")
                
                # Also try to find the file in the items tree
                current_item = None
                for i in range(self.items_tree.topLevelItemCount()):
                    item = self.items_tree.topLevelItem(i)
                    if hasattr(item, 'file_path') and item.file_path == file_path:
                        current_item = item
                        break
                    # Check child items if no match found
                    for j in range(item.childCount()):
                        child = item.child(j)
                        if hasattr(child, 'file_path') and child.file_path == file_path:
                            current_item = child
                            break
                    if current_item:
                        break
                
                # Get Zotero key from the item if available
                if current_item and hasattr(current_item, 'zotero_key'):
                    zotero_key = current_item.zotero_key
                    logger.debug(f"Found Zotero key from item: {zotero_key}")
                
                try:
                    if zotero_key:
                        # Try to find document by Zotero key
                        try:
                            document = PDFDocument.get(PDFDocument.zotero_key == zotero_key)
                            logger.debug(f"Found document by Zotero key: {zotero_key}")
                        except DoesNotExist:
                            logger.debug(f"No document found with Zotero key: {zotero_key}")
                    
                    if not document:
                        # Fall back to file hash if no document found by Zotero key
                        file_hash = calculate_file_hash(file_path)
                        try:
                            document = PDFDocument.get(PDFDocument.file_hash == file_hash)
                            logger.debug(f"Found document by file hash: {file_hash}")
                        except DoesNotExist:
                            logger.debug(f"No document found with file hash: {file_hash}")
                            raise
                        
                except DoesNotExist:
                    logger.error(f"No session found for {file_path}")
                    return
                
            # Get most recent session
            session = (SessionData
                     .select()
                     .where(SessionData.document == document)
                     .order_by(SessionData.created_at.desc())
                     .first())
            
            if not session:
                logger.error(f"No session data found for {file_path}")
                return
            
            # Load session data
            session_data = json.loads(session.session_data)
            self.document_data = session_data.get('document_data', {})
            logger.debug(f"Loaded document data: {self.document_data.keys()}")
            
            # Load page structures
            page_structures = dict(self.document_data.get('page_structures', {}))
            logger.debug(f"Loaded page structures: {page_structures}")
            
            # Open the PDF file
            try:
                self.current_file = file_path
                self.pdf_directory = os.path.dirname(file_path)
                self.doc = fitz.open(file_path)
                self.current_page = session.current_page
                self.pdf_viewer.current_page = self.current_page
                self.pdf_viewer.open_pdf(file_path)
                
                # Update UI
                self.update_navigation()
                
                # Load page structures and show bounding boxes
                for page_num, structure in page_structures.items():
                    page_num = int(page_num)  # Convert string key to int
                    self.document_data['page_structures'][str(page_num)] = structure
                    logger.debug(f"Set structure for page {page_num}")
                
                # Enable bounding boxes if any page has structure
                if page_structures:
                    self.pdf_viewer.toggle_bounding_boxes(True)
                    # Set bounding boxes for current page
                    current_page_boxes = page_structures.get(str(self.current_page), [])
                    if current_page_boxes:
                        self.pdf_viewer.set_bounding_boxes(current_page_boxes.get('structure', {}).get('elements', []))
                
                logger.info(f"Successfully loaded session with {len(self.doc)} pages (Zotero key: {zotero_key})")
                self.status_label.showMessage("Session loaded successfully", 3000)
                
            except Exception as e:
                logger.error(f"Error opening PDF file: {str(e)}")
                QMessageBox.warning(self, "Error", f"Failed to open PDF file: {str(e)}")
                return
            
        except Exception as e:
            logger.error(f"Error loading session: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            QMessageBox.warning(self, "Error", f"Failed to load session: {str(e)}")

    def update_page_display(self):
        """Update the PDF display and text areas with the current page content"""
        logger.debug(f"Updating page display: current page={self.current_page}")
        try:
            if self.doc is None:
                return
            
            # Update PDF display
            self.pdf_viewer.current_page = self.current_page
            self.pdf_viewer.display_page()

            # Get the page structure if it exists
            if str(self.current_page) in self.document_data['page_structures']:
                structure = self.document_data['page_structures'][str(self.current_page)]
                # Extract bounding boxes from structure
                boxes = structure.get('structure', {}).get('elements', [])
                self.pdf_viewer.set_bounding_boxes(boxes)
                logger.debug(f"Setting bounding boxes for page {self.current_page}: {len(boxes)} boxes")
                
                # Update structured content view
                self.structured_view.update_content(self.document_data['page_structures'])
            else:
                self.pdf_viewer.set_bounding_boxes([])
                logger.debug(f"No structure found for page {self.current_page}")
            
            # Update page number display
            self.current_page_input.setText(str(self.current_page + 1))
            self.total_pages_label.setText(f"/ {len(self.doc)}")
            
        except Exception as e:
            logger.error(f"Error updating page display: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            self.status_label.showMessage(f"Error updating display: {str(e)}", 3000)

    def closeEvent(self, event):
        # Save window geometry before closing
        self.settings.setValue("window_geometry", self.saveGeometry())
        super().closeEvent(event)

    def prev_page(self):
        if self.doc and self.current_page > 0:
            self.current_page -= 1
            self.pdf_viewer.current_page = self.current_page
            self.pdf_viewer.display_page()
            self.update_navigation()
            logger.debug(f"Navigated to previous page: {self.current_page + 1}")
            
    def next_page(self):
        if self.doc and self.current_page < len(self.doc) - 1:
            self.current_page += 1
            self.pdf_viewer.current_page = self.current_page
            self.pdf_viewer.display_page()
            self.update_navigation()
            logger.debug(f"Navigated to next page: {self.current_page + 1}")
            
    def update_navigation(self):
        """Update navigation controls and page display"""
        if self.doc:
            self.current_page_input.setText(str(self.current_page + 1))
            self.total_pages_label.setText(f"/ {len(self.doc)}")
            self.prev_button.setEnabled(self.current_page > 0)
            self.next_button.setEnabled(self.current_page < len(self.doc) - 1)
            
            # Update bounding boxes for current page
            if self.pdf_viewer.show_boxes:
                current_page_boxes = self.document_data['page_structures'].get(str(self.current_page), {})
                if current_page_boxes:
                    self.pdf_viewer.set_bounding_boxes(current_page_boxes.get('structure', {}).get('elements', []))
                else:
                    self.pdf_viewer.set_bounding_boxes([])

    def go_to_page(self):
        if not self.doc:
            return
            
        try:
            page_num = int(self.current_page_input.text()) - 1  # Convert to 0-based index
            if 0 <= page_num < len(self.doc):
                self.current_page = page_num
                self.pdf_viewer.current_page = self.current_page
                self.pdf_viewer.display_page()
                self.update_navigation()
                logger.debug(f"Navigated to page: {self.current_page + 1}")
            else:
                QMessageBox.warning(self, "Invalid Page", 
                                  f"Please enter a page number between 1 and {len(self.doc)}")
                self.current_page_input.setText(str(self.current_page + 1))
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid page number")
            self.current_page_input.setText(str(self.current_page + 1))

    def load_zotero_library(self):
        """Load PDFs from Zotero library using collection structure"""
        try:
            # Set wait cursor
            QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
            self.status_label.showMessage("Loading Zotero library...", 0)
            QApplication.processEvents()
            
            # Try default Zotero locations
            zotero_locations = [
                os.path.join(os.getenv('APPDATA', ''), 'Zotero'),  # Default Windows AppData
                os.path.join(os.path.expanduser('~'), 'Zotero'),   # User's home directory
            ]
            
            zotero_dir = None
            for location in zotero_locations:
                db_path = os.path.join(location, 'zotero.sqlite')   
                if os.path.exists(db_path):
                    logger.info(f"Found Zotero database at: {db_path}")
                    zotero_dir = location
                    break
            
            # If not found in default locations, ask user
            if not zotero_dir:
                QApplication.restoreOverrideCursor()  # Temporarily restore cursor for dialog
                zotero_dir = QFileDialog.getExistingDirectory(
                    self,
                    "Select Zotero Directory",
                    os.path.expanduser("~"),
                    QFileDialog.Option.ShowDirsOnly
                )
                if not zotero_dir:
                    return
                QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))  # Set cursor back
            
            # Find the Zotero database
            db_path = os.path.join(zotero_dir, 'zotero.sqlite')
            if not os.path.exists(db_path):
                # Try looking in the profile directory if exists
                profiles_dir = os.path.join(zotero_dir, 'profiles')
                if os.path.exists(profiles_dir):
                    # Look for the first profile directory
                    for profile in os.listdir(profiles_dir):
                        profile_db = os.path.join(profiles_dir, profile, 'zotero.sqlite')
                        if os.path.exists(profile_db):
                            db_path = profile_db
                            break
            
            if not os.path.exists(db_path):
                QMessageBox.warning(self, "Database Not Found", 
                                  f"Could not find Zotero database in {zotero_dir}.\nPlease make sure you selected the correct Zotero directory.")
                return
            
            logger.info(f"Found Zotero database at: {db_path}")
            
            # Connect to the database in read-only mode
            uri = f"file:{db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
            cursor = conn.cursor()
            
            # Get collection structure
            self.status_label.showMessage("Loading collection structure...", 0)
            QApplication.processEvents()
            
            cursor.execute("""
                WITH RECURSIVE collection_tree AS (
                    SELECT 
                        collections.collectionID,
                        collections.collectionName,
                        collections.parentCollectionID,
                        0 as level
                    FROM collections
                    WHERE collections.parentCollectionID IS NULL
                    
                    UNION ALL
                    
                    SELECT 
                        c.collectionID,
                        c.collectionName,
                        c.parentCollectionID,
                        ct.level + 1
                    FROM collections c
                    JOIN collection_tree ct ON c.parentCollectionID = ct.collectionID
                )
                SELECT 
                    collectionID,
                    collectionName,
                    parentCollectionID,
                    level
                FROM collection_tree
                ORDER BY level, collectionName
            """)
            
            collections = cursor.fetchall()
            logger.info(f"Found {len(collections)} collections")
            
            # Get parent items and their collections
            self.status_label.showMessage("Loading parent items...", 0)
            QApplication.processEvents()
            
            cursor.execute("""
                SELECT DISTINCT
                    i.itemID,
                    i.key,
                    idv.value as title,
                    ci.collectionID,
                    it.typeName as itemType,
                    CAST(SUBSTR(
                        (SELECT value 
                         FROM itemData id2 
                         JOIN itemDataValues idv2 ON id2.valueID = idv2.valueID 
                         WHERE id2.itemID = i.itemID 
                         AND id2.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'date')
                         LIMIT 1), 1, 4) AS INTEGER) as year,
                    (SELECT GROUP_CONCAT(c.lastName, '|')
                     FROM (
                         SELECT DISTINCT c.lastName, ic.orderIndex
                         FROM itemCreators ic
                         JOIN creators c ON ic.creatorID = c.creatorID
                         WHERE ic.itemID = i.itemID
                         ORDER BY ic.orderIndex
                     ) as c
                    ) as authors,
                    (SELECT COUNT(*)
                     FROM itemCreators ic
                     WHERE ic.itemID = i.itemID) as author_count
                FROM items i
                JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
                LEFT JOIN itemData id ON i.itemID = id.itemID
                LEFT JOIN itemDataValues idv ON id.valueID = idv.valueID
                LEFT JOIN collectionItems ci ON i.itemID = ci.itemID
                WHERE i.itemID NOT IN (SELECT itemID FROM itemAttachments)
                AND it.typeName NOT IN ('attachment', 'note')
                AND (id.fieldID IS NULL OR id.fieldID = (
                    SELECT fieldID FROM fields WHERE fieldName = 'title'
                ))
                ORDER BY idv.value
            """)
            
            parent_items = cursor.fetchall()
            logger.debug(f"Found {len(parent_items)} parent items")
            
            # Get all PDF attachments (both standalone and child attachments)
            self.status_label.showMessage("Loading PDF attachments...", 0)
            QApplication.processEvents()
            
            cursor.execute("""
                SELECT 
                    a.itemID,
                    a.parentItemID,
                    a.path,
                    COALESCE(
                        (SELECT value 
                         FROM itemData id 
                         JOIN itemDataValues idv ON id.valueID = idv.valueID 
                         WHERE id.itemID = a.itemID 
                         AND id.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'title')
                         LIMIT 1),
                        a.path
                    ) as title,
                    i.key as zotero_key,
                    ci.collectionID,
                    i2.key as parent_key
                FROM itemAttachments a
                JOIN items i ON a.itemID = i.itemID
                LEFT JOIN collectionItems ci ON i.itemID = ci.itemID
                LEFT JOIN items i2 ON a.parentItemID = i2.itemID
                WHERE a.contentType = 'application/pdf'
            """)
            
            attachments = cursor.fetchall()
            logger.info(f"Found {len(attachments)} PDF attachments")
            
            # Create a dictionary of attachments by parent ID for faster lookup
            attachments_by_parent = {}
            standalone_attachments = []
            for attachment in attachments:
                parent_id = attachment[1]  # parentItemID
                if parent_id is not None:
                    if parent_id not in attachments_by_parent:
                        attachments_by_parent[parent_id] = []
                    attachments_by_parent[parent_id].append(attachment)
                else:
                    standalone_attachments.append(attachment)
                logger.debug(f"Attachment {attachment[3]} -> Parent {parent_id} (Key: {attachment[4]}, Parent Key: {attachment[6]})")
            
            # Create collection hierarchy
            self.status_label.showMessage("Creating collection hierarchy...", 0)
            QApplication.processEvents()
            
            collection_map = {}
            root_collections = []
            self.collection_data = {}  # Store item data by collection ID
            
            for collection_id, name, parent_id, level in collections:
                item = QTreeWidgetItem()
                item.setText(0, name)
                item.collection_id = collection_id
                collection_map[collection_id] = item
                self.collection_data[collection_id] = []  # Initialize empty list for items
                
                if parent_id is None:
                    root_collections.append(item)
                else:
                    parent = collection_map.get(parent_id)
                    if parent:
                        parent.addChild(item)
            
            # Configure logging to handle Unicode
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.setStream(sys.stdout)

            # Process items and attachments
            storage_base = os.path.join(zotero_dir, 'storage')
            logger.info(f"Using storage base directory: {storage_base}")
            
            # Function to add PDF item to items tree
            def add_pdf_to_items_tree(zotero_key, display_name, parent_item=None, collection_id=None, parent_key=None):
                storage_dir = os.path.join(storage_base, zotero_key)
                if os.path.exists(storage_dir):
                    # Look for PDF files in the key's directory
                    files = os.listdir(storage_dir)
                    pdf_files = [f for f in files if f.lower().endswith('.pdf')]
                    
                    if pdf_files:
                        # Use the first PDF file found
                        pdf_path = os.path.join(storage_dir, pdf_files[0])
                        
                        # Create PDF item under parent or as top-level item
                        if parent_item is not None:
                            pdf_item = QTreeWidgetItem(parent_item)
                        else:
                            pdf_item = QTreeWidgetItem()
                            self.items_tree.addTopLevelItem(pdf_item)
                        
                        pdf_item.setText(0, display_name)
                        pdf_item.file_path = pdf_path
                        pdf_item.zotero_key = zotero_key  # Store attachment's Zotero key
                        if parent_key:
                            pdf_item.parent_zotero_key = parent_key  # Store parent's Zotero key if available
                        
                        # Add to collection if specified
                        if collection_id and collection_id in self.collection_data:
                            if parent_item is None:  # Only add standalone PDFs to collection items
                                self.collection_data[collection_id].append({
                                    'text': display_name,
                                    'file_path': pdf_path,
                                    'zotero_key': zotero_key,  # Store attachment's Zotero key
                                    'parent_zotero_key': parent_key  # Store parent's Zotero key if available
                                })
                        
                        logger.debug(f"Added PDF: {display_name} from {pdf_path} (Key: {zotero_key}, Parent Key: {parent_key})")
                        return True
                    else:
                        logger.warning(f"No PDF files found in directory: {storage_dir}")
                else:
                    logger.warning(f"Storage directory not found: {storage_dir}")
                return False
            
            # Clear existing items
            self.collections_tree.clear()
            self.items_tree.clear()
            
            # Track processed items and their collections
            processed_items = {}
            
            # Process parent items and their attachments
            total_items = len(parent_items)
            for index, (item_id, key, title, collection_id, item_type, year, authors, author_count) in enumerate(parent_items, 1):
                # Update status every 100 items
                if index % 100 == 0 or index == total_items:
                    self.status_label.showMessage(f"Processing items: {index}/{total_items}", 0)
                    QApplication.processEvents()
                
                if item_id in attachments_by_parent:  # Only process items with PDF attachments
                    # Create display text with metadata
                    display_text = title if title else "Untitled"
                    if authors and year:
                        # Split authors by the separator
                        author_list = authors.split('|')
                        author_count = len(author_list)
                        
                        # Format authors based on count
                        if author_count == 1:
                            author_text = author_list[0]
                        elif author_count == 2:
                            author_text = f"{author_list[0]} and {author_list[1]}"
                        else:
                            author_text = f"{author_list[0]} et al."
                            
                        display_text = f"{display_text} ({author_text}, {year})"
                    elif year:
                        display_text = f"{display_text} ({year})"
                    
                    if item_id not in processed_items:
                        # Create new parent item
                        parent_item = QTreeWidgetItem()
                        parent_item.setText(0, display_text)
                        parent_item.item_id = item_id
                        parent_item.zotero_key = key  # Store parent's Zotero key
                        self.items_tree.addTopLevelItem(parent_item)
                        
                        # Add PDF attachments as children
                        has_pdfs = False
                        for attachment in attachments_by_parent[item_id]:
                            try:
                                zotero_key = attachment[4]  # Zotero key
                                parent_key = attachment[6]  # Parent's Zotero key
                                display_name = attachment[3]
                                if display_name.startswith('storage:'):
                                    display_name = os.path.basename(display_name)
                                
                                if add_pdf_to_items_tree(zotero_key, display_name, parent_item, collection_id, parent_key):
                                    has_pdfs = True
                                    
                            except Exception as e:
                                logger.error(f"Error processing attachment {zotero_key}: {str(e)}")
                        
                        # Store the parent item and its collections
                        processed_items[item_id] = {
                            'parent_item': parent_item,
                            'collections': set([collection_id]) if collection_id else set(),
                            'has_pdfs': has_pdfs,
                            'zotero_key': key  # Store parent's Zotero key
                        }
                    else:
                        # Add collection to existing item
                        if collection_id:
                            processed_items[item_id]['collections'].add(collection_id)
            
            # Process standalone PDF attachments
            total_standalone = len(standalone_attachments)
            for index, attachment in enumerate(standalone_attachments, 1):
                # Update status every 100 items
                if index % 100 == 0 or index == total_standalone:
                    self.status_label.showMessage(f"Processing standalone PDFs: {index}/{total_standalone}", 0)
                    QApplication.processEvents()
                
                try:
                    zotero_key = attachment[4]  # Zotero key
                    display_name = attachment[3]
                    if display_name.startswith('storage:'):
                        display_name = os.path.basename(display_name)
                    
                    add_pdf_to_items_tree(zotero_key, display_name, None, attachment[5])  # attachment[5] is collectionID
                    
                except Exception as e:
                    logger.error(f"Error processing standalone attachment {zotero_key}: {str(e)}")
            
            # Add items to their collections
            self.status_label.showMessage("Organizing collections...", 0)
            QApplication.processEvents()
            
            for item_id, item_data in processed_items.items():
                if item_data['has_pdfs']:
                    for collection_id in item_data['collections']:
                        if collection_id in self.collection_data:
                            self.collection_data[collection_id].append({
                                'text': item_data['parent_item'].text(0),
                                'item_id': item_id,
                                'zotero_key': item_data['zotero_key'],  # Store parent's Zotero key
                                'children': [{
                                    'text': child.text(0),
                                    'file_path': child.file_path,
                                    'zotero_key': child.zotero_key if hasattr(child, 'zotero_key') else None,
                                    'parent_zotero_key': child.parent_zotero_key if hasattr(child, 'parent_zotero_key') else None
                                } for child in [item_data['parent_item'].child(i) for i in range(item_data['parent_item'].childCount())]]
                            })
            
            # Add collections to the collections tree
            for collection in root_collections:
                self.collections_tree.addTopLevelItem(collection)
            
            # Expand the first level of collections
            for i in range(self.collections_tree.topLevelItemCount()):
                self.collections_tree.topLevelItem(i).setExpanded(True)
            
            self.status_label.showMessage("Loaded Zotero library structure", 3000)
            logger.info("Loaded Zotero library structure")
            
        except Exception as e:
            logger.error(f"Error loading Zotero library: {str(e)}")
            QMessageBox.warning(self, "Error", f"Could not load Zotero library:\n{str(e)}")
            
        finally:
            # Restore cursor
            QApplication.restoreOverrideCursor()
            self.ensure_normal_cursor()

    def batch_analyze(self):
        """Analyze all PDF files in the items tree"""
        try:
            # Create dialog for analysis options
            dialog = QDialog(self)
            dialog.setWindowTitle("Batch Analysis Options")
            layout = QVBoxLayout()

            # Add radio buttons for analysis scope
            scope_group = QButtonGroup(dialog)
            all_pdfs_radio = QRadioButton("Analyze all PDFs in library")
            selected_collection_radio = QRadioButton("Analyze PDFs in selected collection")
            all_pdfs_radio.setChecked(True)
            scope_group.addButton(all_pdfs_radio)
            scope_group.addButton(selected_collection_radio)
            layout.addWidget(all_pdfs_radio)
            layout.addWidget(selected_collection_radio)

            # Add collection tree widget (initially disabled)
            collection_tree = QTreeWidget()
            collection_tree.setHeaderHidden(True)
            collection_tree.setEnabled(False)
            layout.addWidget(collection_tree)

            # Populate collection tree
            def add_collection_to_tree(parent_item, collection):
                item = QTreeWidgetItem(parent_item)
                item.setText(0, collection.text(0))
                item.collection_id = collection.collection_id
                for i in range(collection.childCount()):
                    add_collection_to_tree(item, collection.child(i))

            for i in range(self.collections_tree.topLevelItemCount()):
                collection = self.collections_tree.topLevelItem(i)
                add_collection_to_tree(collection_tree, collection)

            # Expand first level of collections
            for i in range(collection_tree.topLevelItemCount()):
                collection_tree.topLevelItem(i).setExpanded(True)

            # Connect radio buttons to enable/disable collection tree
            def update_collection_tree():
                collection_tree.setEnabled(selected_collection_radio.isChecked())
            all_pdfs_radio.toggled.connect(update_collection_tree)
            selected_collection_radio.toggled.connect(update_collection_tree)

            # Add buttons
            button_box = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | 
                QDialogButtonBox.StandardButton.Cancel
            )
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)

            dialog.setLayout(layout)

            # Show dialog and get user choice
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return

            # Get selected collection ID if applicable
            selected_collection_id = None
            if selected_collection_radio.isChecked():
                selected_item = collection_tree.currentItem()
                if selected_item:
                    selected_collection_id = selected_item.collection_id
                    logger.info(f"Selected collection for analysis: {selected_item.text(0)}")
                else:
                    QMessageBox.warning(self, "No Collection Selected", 
                                      "Please select a collection to analyze.")
                    return

            # Set wait cursor
            QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
            self.status_label.showMessage("Starting batch analysis...", 0)
            QApplication.processEvents()

            total_files = 0
            analyzed_files = 0
            failed_files = 0

            # Store current file to restore later
            previous_file = self.current_file if hasattr(self, 'current_file') else None
            previous_doc = self.doc if hasattr(self, 'doc') else None

            def process_pdf_item(item):
                """Process a single PDF item"""
                nonlocal analyzed_files, failed_files
                
                # Select and scroll to the item
                self.items_tree.setCurrentItem(item)
                self.items_tree.scrollToItem(item)
                QApplication.processEvents()

                try:
                    # Open the PDF file and show first page
                    self.current_file = item.file_path
                    self.pdf_directory = os.path.dirname(item.file_path)
                    self.doc = fitz.open(item.file_path)
                    self.current_page = 0
                    self.pdf_viewer.current_page = 0
                    self.pdf_viewer.open_pdf(item.file_path)
                    self.update_navigation()
                    QApplication.processEvents()

                    # Analyze the PDF
                    if self.analyze_pdf(item.file_path):
                        analyzed_files += 1
                        # Update item text to show it's analyzed without changing the filename
                        original_text = item.text(0)
                        if not original_text.endswith(" (Analyzed)"):
                            item.setText(0, f"{original_text} (Analyzed)")
                        # Set text color to green to indicate success
                        item.setForeground(0, Qt.GlobalColor.darkGreen)
                    else:
                        failed_files += 1
                        # Update item text to show failure without changing the filename
                        original_text = item.text(0)
                        if not original_text.endswith(" (Failed)"):
                            item.setText(0, f"{original_text} (Failed)")
                        # Set text color to red to indicate failure
                        item.setForeground(0, Qt.GlobalColor.red)

                    # Close the document
                    self.doc.close()

                except Exception as e:
                    logger.error(f"Error analyzing {item.file_path}: {str(e)}")
                    failed_files += 1
                    # Update item text to show error without changing the filename
                    original_text = item.text(0)
                    if not original_text.endswith(" (Error)"):
                        item.setText(0, f"{original_text} (Error)")
                    # Set text color to red to indicate error
                    item.setForeground(0, Qt.GlobalColor.red)

                    # Make sure to close the document if it's open
                    if hasattr(self, 'doc') and self.doc:
                        try:
                            self.doc.close()
                        except:
                            pass

            def process_collection(collection):
                """Process a collection and its items recursively"""
                nonlocal total_files
                
                # Select the collection to populate items tree
                self.collections_tree.setCurrentItem(collection)
                self.collection_clicked(collection, 0)
                QApplication.processEvents()
                
                # Process each item in the items tree
                for i in range(self.items_tree.topLevelItemCount()):
                    item = self.items_tree.topLevelItem(i)
                    
                    # Check if item is a PDF
                    if hasattr(item, 'file_path') and item.file_path.lower().endswith('.pdf'):
                        total_files += 1
                        # Update status
                        self.status_label.showMessage(
                            f"Analyzing {os.path.basename(item.file_path)} "
                            f"in {collection.text(0)}...", 0)
                        QApplication.processEvents()
                        
                        process_pdf_item(item)
                    
                    # Check children for PDFs
                    for j in range(item.childCount()):
                        child = item.child(j)
                        if hasattr(child, 'file_path') and child.file_path.lower().endswith('.pdf'):
                            total_files += 1
                            # Update status
                            self.status_label.showMessage(
                                f"Analyzing {os.path.basename(child.file_path)} "
                                f"in {collection.text(0)}...", 0)
                            QApplication.processEvents()
                            
                            process_pdf_item(child)
                
                # Process subcollections recursively
                for i in range(collection.childCount()):
                    subcollection = collection.child(i)
                    process_collection(subcollection)

            # Process collections
            if selected_collection_id:
                # Find and select the specific collection
                selected_collection = None
                def find_collection(item, target_id):
                    if hasattr(item, 'collection_id') and item.collection_id == target_id:
                        return item
                    for i in range(item.childCount()):
                        result = find_collection(item.child(i), target_id)
                        if result:
                            return result
                    return None

                # Search through all collections recursively
                for i in range(self.collections_tree.topLevelItemCount()):
                    item = self.collections_tree.topLevelItem(i)
                    selected_collection = find_collection(item, selected_collection_id)
                    if selected_collection:
                        break

                if selected_collection:
                    process_collection(selected_collection)
                    logger.info(f"Processed selected collection: {selected_collection.text(0)}")
            else:
                # Process all collections
                for i in range(self.collections_tree.topLevelItemCount()):
                    collection = self.collections_tree.topLevelItem(i)
                    process_collection(collection)
                    logger.info(f"Processed collection: {collection.text(0)}")

            if total_files == 0:
                QMessageBox.information(self, "No PDF Files", 
                                      "No PDF files found to analyze.")
                return

            # Restore previous file if there was one
            if previous_file and os.path.exists(previous_file):
                try:
                    self.current_file = previous_file
                    self.pdf_directory = os.path.dirname(previous_file)
                    self.doc = fitz.open(previous_file)
                    self.pdf_viewer.open_pdf(previous_file)
                    self.update_page_display()
                except Exception as e:
                    logger.error(f"Error restoring previous file: {str(e)}")

            # Show completion message
            msg = f"Batch analysis completed. Analyzed: {analyzed_files}, Failed: {failed_files}, Total: {total_files}"
            self.status_label.showMessage(msg, 5000)
            QMessageBox.information(self, "Batch Analysis Complete", msg)

        except Exception as e:
            logger.error(f"Error in batch analysis: {str(e)}")
            QMessageBox.warning(self, "Error", f"Error during batch analysis: {str(e)}")

        finally:
            # Restore cursor
            QApplication.restoreOverrideCursor()
            self.ensure_normal_cursor()

def main():
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        logger.critical(f"Application error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
'''
Build command:
pyinstaller --name "PDFRefinery_v0.0.1.exe" --add-data "icons/*.png:icons" --onefile --noconsole PDFRefinery.py
'''