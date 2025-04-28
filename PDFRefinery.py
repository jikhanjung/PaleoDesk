import sys
import logging
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QFileDialog, QToolBar,
                           QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                           QDialogButtonBox, QMenuBar, QMenu, QMessageBox,
                           QTreeWidget, QTreeWidgetItem, QDockWidget, QSplitter,
                           QComboBox, QRadioButton, QButtonGroup, QTabWidget,
                           QGridLayout, QStackedWidget, QFormLayout, QTextEdit,
                           QScrollArea, QSizePolicy, QLayout, QListWidget, QListWidgetItem,
                           QFrame, QStatusBar, QProgressBar, QCheckBox, QGroupBox, QToolTip)
from PyQt6.QtCore import Qt, QPoint, QSettings, QSize, QRect, QRectF, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QPainter, QAction, QCursor, QIcon, QPen, QColor
import fitz  # PyMuPDF
import datetime
import os
import requests
import json
import sqlite3
import shutil
from peewee import DoesNotExist, chunked
from PDFModels import (db, PDFDocument, PageAnalysis, SessionData, StructuredElement, init_database,
                      calculate_file_hash, DEFAULT_LOG_DIRECTORY, COMPANY_NAME,
                      PROGRAM_NAME, DEFAULT_DB_DIRECTORY, DB_PATH)
import hashlib
import uuid
import copy

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
    # Add signal for wheel events
    wheel_scrolled = pyqtSignal(int)  # Signal to emit the scroll amount
    
    def __init__(self):
        super().__init__()
        self.pixmap = None
        self.current_page = 0
        self.zoom = 1.0
        self.drag_start = None
        self.drag_pos = None
        self.bounding_boxes = {}
        self.show_bounding_boxes = False
        self.pdf_document= None
        self.total_pages = 0
        self.initial_load_pages = 3  # Number of pages to load initially
        self.loaded_pages = set()  # Track which pages are loaded
        self.page_pixmaps = {}  # Cache for page pixmaps
        self.page_loading = False  # Flag to prevent multiple simultaneous loads
        self.pan_offset = QPoint(0, 0)  # Add pan offset
        self.last_pan_pos = None  # Add last pan position
        self.current_page_width = 0  # Add current page width
        self.current_page_height = 0  # Add current page height
        self.hovered_box = None  # Track currently hovered bounding box
        self.hovered_box_page = None  # Track page number of hovered box
        # Add these new instance variables
        self.dragging_box = None  # Currently dragged box
        self.drag_start_pos = None  # Mouse position when drag started
        self.original_box_coords = None  # Original coordinates of the box being dragged
        self.current_box_page = None  # Page number of the box being dragged
        self.resize_edges = None  # Which edges are being resized (e.g., 'n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw')
        self.edge_threshold = 10  # Pixels from edge to trigger resize cursor
        
        # Add element creation variables
        self.creating_element = False
        self.element_start_pos = None
        self.element_current_pos = None
        self.element_type = 'text'  # Default type
        
        # Set up the widget
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # Create zoom buttons
        self.create_zoom_buttons()
        
        # Set minimum size
        self.setMinimumSize(400, 600)
        
        # Set background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), Qt.GlobalColor.white)
        self.setPalette(palette)
        
    def contextMenuEvent(self, event):
        """Handle right-click context menu"""
        # Get the main window instance
        main_window = self.window()
        if not main_window or not hasattr(main_window, 'document_data'):
            return
            
        # Use the hovered box information from mouseMoveEvent
        if self.hovered_box and self.hovered_box_page is not None:
            clicked_box = self.hovered_box
            clicked_box_id = clicked_box.get('id')
            logger.info(f"clicked_box: {clicked_box}, clicked_box_id: {clicked_box_id}, clicked_box_page: {self.hovered_box_page}")
            
            # Create context menu
            menu = QMenu(self)
            
            # Add "Change Block Type" submenu
            change_type_menu = menu.addMenu("Change Block Type")
            
            # Add block type actions
            block_types = ['text', 'figure', 'table', 'picture', 'caption']
            for block_type in block_types:
                action = change_type_menu.addAction(block_type.capitalize())
                action.triggered.connect(lambda checked, t=block_type: self._change_block_type(clicked_box_id, self.hovered_box_page, t))
            
            # Show menu at cursor position
            menu.exec(event.globalPos())
    
    def _change_block_type(self, box_id, page_num, new_type):
        """Change the type of a block"""
        # Get the main window instance
        main_window = self.window()
        if not main_window or not hasattr(main_window, 'document_data'):
            return
            
        # Update the element type in document_data
        page_key = str(page_num)
        if page_key in main_window.document_data['page_structures']:
            page_structure = main_window.document_data['page_structures'][page_key]
            if 'structure' in page_structure and 'elements' in page_structure['structure']:
                elements = page_structure['structure']['elements']
                for element in elements:
                    logger.info(f"page: {page_num}, element_id: {element.get('id')}, box_id: {box_id}")
                    if element.get('id') == box_id:
                        old_type = element.get('category', '')
                        element['category'] = new_type
                        logger.info(f"Changed element {box_id} type from {old_type} to {new_type}")
                        
                        # Save the element to database
                        self.save_element(page_num, box_id)
                        
                        # Update the structured content view
                        if hasattr(main_window, 'structured_view'):
                            main_window.structured_view.update_content(main_window.document_data['page_structures'])
                            #main_window.save_session()
                        break

    def create_zoom_buttons(self):
        """Create zoom control buttons"""
        # Create container widget for buttons
        self.zoom_widget = QWidget(self)
        self.zoom_widget.setObjectName("zoomWidget")
        self.zoom_widget.setStyleSheet("""
            QWidget#zoomWidget {
                background-color: rgba(255, 255, 255, 180);
                border-radius: 5px;
            }
            QPushButton {
                background-color: white;
                border: 1px solid #ccc;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
        """)
        
        # Create layout for buttons
        zoom_layout = QHBoxLayout(self.zoom_widget)
        zoom_layout.setContentsMargins(5, 5, 5, 5)
        zoom_layout.setSpacing(5)
        
        # Create zoom out button
        self.zoom_out_btn = QPushButton("-")
        self.zoom_out_btn.setFixedSize(30, 30)
        self.zoom_out_btn.clicked.connect(lambda: self.set_zoom(self.zoom / 1.1))
        zoom_layout.addWidget(self.zoom_out_btn)
        
        # Create zoom in button
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setFixedSize(30, 30)
        self.zoom_in_btn.clicked.connect(lambda: self.set_zoom(self.zoom * 1.1))
        zoom_layout.addWidget(self.zoom_in_btn)
        
        # Position the zoom widget
        self.zoom_widget.setFixedSize(80, 40)
        self.zoom_widget.move(self.width() - 90, 10)  # Position in top-right corner

    def open_pdf(self, file_path):
        """Open a PDF file and load initial pages"""
        try:
            self.pdf_document= fitz.open(file_path)
            self.total_pages = len(self.pdf_document)
            self.current_page = 0
            self.loaded_pages.clear()
            self.page_pixmaps.clear()
            
            # Load initial pages
            self.load_initial_pages()
            
            # Update the display
            self.update_current_page()
            self.update()
            
            logger.info(f"Opened PDF with {self.total_pages} pages")
            return True
        except Exception as e:
            logger.error(f"Error opening PDF: {str(e)}")
            return False

    def load_initial_pages(self):
        """Load the initial set of pages"""
        try:
            # Load first few pages
            for page_num in range(min(self.initial_load_pages, self.total_pages)):
                self.load_page(page_num)
            
            # Start loading next set of pages in background
            self.load_next_pages()
        except Exception as e:
            logger.error(f"Error loading initial pages: {str(e)}")

    def load_page(self, page_num):
        """Load a single page and cache its pixmap"""
        if page_num in self.loaded_pages or page_num >= self.total_pages:
            return
        
        try:
            page = self.pdf_document[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(self.zoom, self.zoom))
            img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(img)
            self.page_pixmaps[page_num] = {
                'pixmap': pixmap,
                'width': pix.width,
                'height': pix.height
            }
            self.loaded_pages.add(page_num)
            logger.debug(f"Loaded page {page_num}")
        except Exception as e:
            logger.error(f"Error loading page {page_num}: {str(e)}")

    def load_next_pages(self):
        """Load next set of pages in background"""
        if self.page_loading or not self.pdf_document:
            return
        
        self.page_loading = True
        try:
            # Find the highest loaded page number
            max_loaded = max(self.loaded_pages) if self.loaded_pages else -1
            
            # Load next set of pages
            next_pages = range(max_loaded + 1, min(max_loaded + self.initial_load_pages + 1, self.total_pages))
            for page_num in next_pages:
                self.load_page(page_num)
            
            # Update display if current page was loaded
            if self.current_page in self.loaded_pages:
                self.update_current_page()
                self.update()
        finally:
            self.page_loading = False

    def update_current_page(self):
        """Update the current page display"""
        if not self.pdf_documentor or self.current_page >= self.total_pages:
            self.pixmap = None
            self.current_page_width = 0
            self.current_page_height = 0
            return
        
        # Load the page if not already loaded
        if self.current_page not in self.loaded_pages:
            self.load_page(self.current_page)
        
        # Get the page pixmap from cache
        if self.current_page in self.page_pixmaps:
            page_data = self.page_pixmaps[self.current_page]
            self.pixmap = page_data['pixmap']
            self.current_page_width = page_data['width']
            self.current_page_height = page_data['height']
            
            # Start loading next pages if we're near the end of loaded pages
            if self.current_page >= max(self.loaded_pages) - 1:
                self.load_next_pages()
        else:
            self.pixmap = None
            self.current_page_width = 0
            self.current_page_height = 0

    def scroll_to_page(self, page_num):
        """Scroll to a specific page"""
        if 0 <= page_num < self.total_pages:
            self.current_page = page_num
            self.update_current_page()
            self.update()

    def set_current_page(self, page_num):
        """Set the current page number"""
        if 0 <= page_num < self.total_pages:
            self.current_page = page_num
            self.update_current_page()
            self.update()

    def update_current_page_from_scroll(self, scroll_value):
        """Update current page based on scroll position"""
        if not self.pdf_document:
            return
            
        # Calculate viewport center
        viewport_height = self.height()
        viewport_center = scroll_value + viewport_height / 2
        
        # Get page heights from loaded pages
        page_heights = [page_data['height'] for page_data in self.page_pixmaps.values()]
        logger.debug(f"Page heights: {page_heights}")
        
        # Find the page that contains the viewport center
        current_height = 0
        for i, height in enumerate(page_heights):
            page_bottom = current_height + height
            logger.debug(f"Page {i} range: {current_height} to {page_bottom}")
            
            if current_height <= viewport_center < page_bottom:
                if self.current_page != i:
                    self.current_page = i
                    logger.debug(f"Current page updated to {i + 1}")
                break
            current_height = page_bottom

    def paintEvent(self, event):
        """Handle painting of the widget"""
        if not self.page_pixmaps:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Define category colors with alpha
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
        
        # Draw pages
        current_y = 0
        for page_num, page_data in self.page_pixmaps.items():
            pixmap = page_data['pixmap']
            height = page_data['height']
            
            # Calculate scaling to fit width while maintaining aspect ratio
            scale = self.width() / pixmap.width()
            scaled_height = int(height * scale)  # Convert to int
            
            # Draw the page
            painter.drawPixmap(0, int(current_y), pixmap.scaled(self.width(), scaled_height, 
                                                         Qt.AspectRatioMode.KeepAspectRatio,
                                                         Qt.TransformationMode.SmoothTransformation))
            
            # Draw bounding boxes if enabled
            if self.show_bounding_boxes and self.bounding_boxes:
                # Get boxes for current page
                page_boxes = self.bounding_boxes.get(page_num, [])
                for element in page_boxes:
                    if 'coordinates' in element:
                        coords = element['coordinates']
                        if len(coords) == 4:  # Ensure we have all 4 corners
                            # Scale coordinates to match displayed page
                            x1 = int(coords[0]['x'] * self.width())
                            y1 = int(coords[0]['y'] * scaled_height + current_y)
                            x2 = int(coords[2]['x'] * self.width())
                            y2 = int(coords[2]['y'] * scaled_height + current_y)
                            
                            # Get category and corresponding color
                            category = element.get('category', 'text').lower()
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
            
            current_y += scaled_height
            
        # Draw element creation rectangle if in progress
        if self.creating_element and self.element_start_pos is not None and self.element_current_pos is not None:
            painter.setPen(QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.DashLine))
            x1 = min(self.element_start_pos.x(), self.element_current_pos.x())
            y1 = min(self.element_start_pos.y(), self.element_current_pos.y())
            x2 = max(self.element_start_pos.x(), self.element_current_pos.x())
            y2 = max(self.element_start_pos.y(), self.element_current_pos.y())
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

    def set_bounding_boxes(self, boxes):
        """Set bounding boxes for all pages"""
        if not boxes:
            return
            
        # Clear existing boxes
        self.bounding_boxes = {}
        
        # Get the main window instance
        main_window = self.window()
        if not hasattr(main_window, 'document_data'):
            return
            
        # Get all page structures
        page_structures = main_window.document_data.get('page_structures', {})
        
        # Set boxes for each page
        for page_num, structure in page_structures.items():
            page_boxes = structure.get('structure', {}).get('elements', [])
            if page_boxes:
                self.bounding_boxes[int(page_num)] = page_boxes
                logger.debug(f"Set {len(page_boxes)} bounding boxes for page {int(page_num) + 1}")
        
        self.update()

    def toggle_bounding_boxes(self, show):
        """Toggle bounding box display"""
        self.show_bounding_boxes = show
        self.update()

    def set_zoom(self, new_zoom):
        """Set the zoom level"""
        if 0.1 <= new_zoom <= 5.0:  # Limit zoom range
            self.zoom = new_zoom
            # Clear page cache when zoom changes
            self.page_pixmaps.clear()
            self.loaded_pages.clear()
            # Reload current page with new zoom
            self.update_current_page()
            self.update()
        
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if self.creating_element and event.button() == Qt.MouseButton.LeftButton:
            self.element_start_pos = event.pos()
            self.element_current_pos = event.pos()
            return
            
        if event.button() == Qt.MouseButton.LeftButton:
            # If we're over a box, start dragging or resizing
            if self.dragging_box is not None:
                self.drag_start_pos = event.pos()
                # Store original coordinates and element id
                self.original_box_coords = self.dragging_box['coordinates'].copy()
                self.dragging_box_id = self.dragging_box.get('id')  # Store the element id
            else:
                # Normal panning behavior
                self.last_pan_pos = event.pos()

    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
        if self.creating_element and self.element_start_pos is not None:
            self.element_current_pos = event.pos()
            self.update()  # Trigger repaint to show the rectangle
            return
            
        main_window = self.window()
        if not main_window:
            return

        if self.drag_start_pos and self.dragging_box and event.buttons() & Qt.MouseButton.LeftButton:
            # Calculate the movement delta in viewport coordinates
            delta_x = event.pos().x() - self.drag_start_pos.x()
            delta_y = event.pos().y() - self.drag_start_pos.y()
            
            # Convert delta to PDF coordinates
            widget_width = self.width()
            first_page_width = next((self.page_pixmaps[i]['width'] for i in range(len(self.page_pixmaps)) if i in self.page_pixmaps), None)
            if not first_page_width:
                return
                
            width_ratio = widget_width / (first_page_width * self.zoom)
            
            # Get page dimensions
            page_width = self.page_pixmaps[self.current_box_page]['width']
            page_height = self.page_pixmaps[self.current_box_page]['height']
            
            # Convert movement to normalized coordinates
            screen_page_width = page_width * self.zoom * width_ratio
            screen_page_height = page_height * self.zoom * width_ratio
            
            delta_x_norm = delta_x / screen_page_width
            delta_y_norm = delta_y / screen_page_height

            if self.resize_edges:
                # Resizing - update only the edges being dragged
                coords = self.dragging_box['coordinates']
                if 'n' in self.resize_edges:
                    coords[0]['y'] = self.original_box_coords[0]['y'] + delta_y_norm
                    coords[1]['y'] = self.original_box_coords[1]['y'] + delta_y_norm
                if 's' in self.resize_edges:
                    coords[2]['y'] = self.original_box_coords[2]['y'] + delta_y_norm
                    coords[3]['y'] = self.original_box_coords[3]['y'] + delta_y_norm
                if 'w' in self.resize_edges:
                    coords[0]['x'] = self.original_box_coords[0]['x'] + delta_x_norm
                    coords[3]['x'] = self.original_box_coords[3]['x'] + delta_x_norm
                if 'e' in self.resize_edges:
                    coords[1]['x'] = self.original_box_coords[1]['x'] + delta_x_norm
                    coords[2]['x'] = self.original_box_coords[2]['x'] + delta_x_norm
            else:
                # Normal dragging - move all points
                for i in range(4):
                    self.dragging_box['coordinates'][i]['x'] = self.original_box_coords[i]['x'] + delta_x_norm
                    self.dragging_box['coordinates'][i]['y'] = self.original_box_coords[i]['y'] + delta_y_norm
            
            # Update drag start position to current position for smooth continuous dragging
            self.drag_start_pos = event.pos()
            
            self.update()  # Redraw the view
        elif self.last_pan_pos and event.buttons() & Qt.MouseButton.LeftButton:
            # Normal panning behavior
            delta = event.pos() - self.last_pan_pos
            self.pan_offset += delta
            self.last_pan_pos = event.pos()
            self.update()
        else:
            # Check if mouse is over a bounding box
            self._check_bounding_box_hover(event.pos())

    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if self.creating_element and event.button() == Qt.MouseButton.LeftButton and self.element_start_pos is not None:
            # Get the main window instance
            main_window = self.window()
            if not main_window or not hasattr(main_window, 'document_data'):
                logger.error("No main window or document data found")
                return
                
            # Calculate the rectangle coordinates
            x1 = min(self.element_start_pos.x(), self.element_current_pos.x())
            y1 = min(self.element_start_pos.y(), self.element_current_pos.y())
            x2 = max(self.element_start_pos.x(), self.element_current_pos.x())
            y2 = max(self.element_start_pos.y(), self.element_current_pos.y())
            
            # Get current page height from page_pixmaps
            if self.current_cursor_page in self.page_pixmaps:
                page_height = self.page_pixmaps[self.current_cursor_page]['height']
                
                # Calculate total height of pages before current page
                total_height_before = 0
                for i in range(self.current_cursor_page):
                    if i in self.page_pixmaps:
                        total_height_before += self.page_pixmaps[i]['height']
                
                # Calculate scaling to fit width while maintaining aspect ratio
                scale = self.width() / self.page_pixmaps[self.current_cursor_page]['width']
                scaled_height = int(page_height * scale)  # Convert to int
                
                # Adjust y coordinates by subtracting the height of previous pages and accounting for scaling
                y1_adjusted = (y1 - total_height_before) / (scale * self.zoom)
                y2_adjusted = (y2 - total_height_before) / (scale * self.zoom)
                
                # Convert screen coordinates to relative coordinates
                rel_x1 = x1 / (self.width() * self.zoom)
                rel_y1 = y1_adjusted / page_height
                rel_x2 = x2 / (self.width() * self.zoom)
                rel_y2 = y2_adjusted / page_height
                
                logger.info(f"Creating new element on page {self.current_cursor_page}")
                logger.info(f"Screen coordinates: ({x1}, {y1}) to ({x2}, {y2})")
                logger.info(f"Page height: {page_height}")
                logger.info(f"Total height before current page: {total_height_before}")
                logger.info(f"Scale: {scale}, Zoom: {self.zoom}")
                logger.info(f"Adjusted y coordinates: ({y1_adjusted}, {y2_adjusted})")
                logger.info(f"Relative coordinates: ({rel_x1:.3f}, {rel_y1:.3f}) to ({rel_x2:.3f}, {rel_y2:.3f})")
                
                # Get current page elements to determine new element ID
                page_key = str(self.current_cursor_page)
                if page_key in main_window.document_data['page_structures']:
                    page_structure = main_window.document_data['page_structures'][page_key]
                    if 'structure' in page_structure and 'elements' in page_structure['structure']:
                        elements = page_structure['structure']['elements']
                        # show elements' page, id and category
                        logger.info(f"Elements on current page: {[{'page': page_key, 'id': e['id'], 'category': e['category']} for e in elements]}")
                        
                        new_id = str(len(elements))  # Use element count as ID
                        logger.info(f"Current page has {len(elements)} elements, new element ID will be {new_id}")
                        
                        # Create new element with normalized coordinates
                        new_element = {
                            'id': new_id,
                            'category': self.element_type,
                            'coordinates': [
                                {'x': rel_x1, 'y': rel_y1},  # top-left
                                {'x': rel_x2, 'y': rel_y1},  # top-right
                                {'x': rel_x2, 'y': rel_y2},  # bottom-right
                                {'x': rel_x1, 'y': rel_y2}   # bottom-left
                            ],
                            'content': {'text': ''},
                            'attributes': {
                                'page_width': self.current_page_width,
                                'page_height': self.current_page_height
                            }
                        }
                        logger.info(f"Created new element: {new_element}")
                        
                        # Add element to current page
                        elements.append(new_element)
                        logger.info(f"Added element to page structure, now has {len(elements)} elements")
                        
                        # Save the new element to database
                        self.save_element(self.current_cursor_page, new_id)
                        
                        # Update bounding boxes
                        if self.current_cursor_page not in self.bounding_boxes:
                            self.bounding_boxes[self.current_cursor_page] = []
                        #self.bounding_boxes[self.current_cursor_page].append(new_element)
                        #logger.info(f"Added element to bounding boxes for page {self.current_cursor_page}")
                        
                        # Update structured content view
                        if hasattr(main_window, 'structured_view'):
                            logger.info("Updating structured content view")
                            main_window.structured_view.update_content(main_window.document_data['page_structures'])
                            logger.info("Saving session")
                            #main_window.save_session()  # Auto-save after adding element
                        else:
                            logger.warning("No structured view found in main window")
                else:
                    logger.error(f"Page {page_key} not found in page structures")
            else:
                logger.error(f"Page {self.current_cursor_page} not found in page_pixmaps")
            
            # Reset element creation state
            self.creating_element = False
            self.element_start_pos = None
            self.element_current_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
            
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_pan_pos = None
            if self.dragging_box:
                logger.debug(f"Box release - Page: {self.current_box_page}, Box ID: {self.dragging_box_id}")
                logger.debug(f"New coordinates: {self.dragging_box['coordinates']}")
                
                # Update the page structure with new box coordinates
                if self.current_box_page is not None and self.current_box_page in self.bounding_boxes:
                    # Find and update the box in the page structure
                    page_boxes = self.bounding_boxes[self.current_box_page]
                    for i, box in enumerate(page_boxes):
                        if box is self.dragging_box:  # Compare by reference
                            # Update the box coordinates in the page structure
                            page_boxes[i] = self.dragging_box
                            logger.debug(f"Updated box in bounding_boxes[{self.current_box_page}][{i}]")
                            break
                    
                    # Update the main window's page structure if available
                    # Find the main window by traversing up the widget hierarchy
                    parent = self.parent()
                    while parent is not None:
                        if isinstance(parent, MainWindow):
                            main_window = parent
                            break
                        parent = parent.parent()
                    
                    if parent is None:
                        logger.warning("Could not find MainWindow in widget hierarchy")
                        return

                    logger.debug(f"Found main window: {main_window}")
                    if hasattr(main_window, 'document_data'):
                        logger.debug("Found document_data in main window")
                        if 'page_structures' in main_window.document_data:
                            logger.debug(f"Found page_structures in document_data")
                            page_key = str(self.current_box_page)
                            if page_key in main_window.document_data['page_structures']:
                                logger.debug(f"Found page {page_key} in page_structures")
                                page_structure = main_window.document_data['page_structures'][page_key]
                                if 'structure' in page_structure and 'elements' in page_structure['structure']:
                                    elements = page_structure['structure']['elements']
                                    logger.debug(f"Found {len(elements)} elements in page structure")
                                    # Find and update the corresponding element by id
                                    for element in elements:
                                        if element.get('id') == self.dragging_box_id:
                                            logger.debug(f"Found matching element with id {self.dragging_box_id}")
                                            # Update coordinates
                                            element['coordinates'] = self.dragging_box['coordinates']
                                            
                                            # Save the updated element to database
                                            self.save_element(self.current_box_page, self.dragging_box_id)
                                            
                                            # Update the structured content view
                                            if hasattr(main_window, 'structured_view'):
                                                logger.debug("Updating structured content view")
                                                main_window.structured_view.update_content(main_window.document_data['page_structures'])
                                                #main_window.save_session()
                                                logger.debug("Structured content view update called")
                                            else:
                                                logger.warning("No structured_view found in main window")
                                            break
                                    else:
                                        logger.warning(f"No element found with id {self.dragging_box_id}")
                                else:
                                    logger.warning("No 'structure' or 'elements' in page_structure")
                            else:
                                logger.warning(f"Page {page_key} not found in page_structures")
                        else:
                            logger.warning("No page_structures found in document_data")
                    else:
                        logger.warning("No document_data found in main window")
                
                # Reset dragging state
                self.drag_start_pos = None
                self.original_box_coords = None
                self.dragging_box_id = None
                self.resize_edges = None
                # Keep dragging_box set so we maintain hover state
                
                # Force a redraw
                self.update()

    def save_element(self, page_num, element_id):
        """Save the element to database and update the page structure"""
        # Get the element from the page structure
        main_window = self.window()
        
        page_structure = main_window.document_data['page_structures'][str(page_num)]
        element = page_structure['structure']['elements'][int(element_id)]
        logger.info(f"Saving element {element} from page {page_num}")

        pdf_path = main_window.current_file_path
        if not pdf_path:
            logger.warning("No current document to save element to")
            return
        
        # Get the document from the database
        zotero_key = None
        file_hash = None
        zotero_key = main_window.get_zotero_key_for_current_file()
        if not zotero_key:
            logger.debug("No Zotero key found for current file, using file hash")
            file_hash = main_window.calculate_file_hash(main_window.current_file_path)
        else:
            file_hash = None
        document = main_window.load_document_from_database(main_window.current_file_path, zotero_key, file_hash)
        if not document:
            logger.error("No document found in database")
            return

        try:
            # Create or update the structured element
            structured_element = StructuredElement.get_or_create(
                document=document,
                page_number=page_num,
                element_id=str(element_id),
                defaults={
                    'element_type': element.get('category', 'unknown'),
                    'coordinates': json.dumps(element.get('coordinates', [])),
                    'content': json.dumps(element.get('content', {})),
                    'caption': json.dumps(element.get('caption', {})),
                    'metadata': json.dumps(element.get('metadata', {})),
                    'created_at': datetime.datetime.now(),
                    'updated_at': datetime.datetime.now()
                }
            )[0]

            old_type = structured_element.element_type
            new_type = element.get('category', 'unknown')
            
            # Update the element if it already existed
            structured_element.element_type = element.get('category', 'unknown')
            structured_element.coordinates = json.dumps(element.get('coordinates', []))
            structured_element.content = json.dumps(element.get('content', {}))
            structured_element.caption = json.dumps(element.get('caption', {}))
            structured_element.metadata = json.dumps(element.get('metadata', {}))
            structured_element.updated_at = datetime.datetime.now()
            structured_element.save()

            #logger.info(f"Saved row_id {structured_element.id} for element {element_id} from page {page_num} to database {old_type} -> {new_type}")
            
        except Exception as e:
            logger.error(f"Error saving element to database: {str(e)}")
            QMessageBox.warning(self, "Save Error", 
                f"Error saving element to database: {str(e)}")

    def update_current_page(self):
        """Update current page based on visible area"""
        if not self.page_pixmaps:
            return
            
        # Calculate which page is most visible in the viewport
        viewport_center = -self.pan_offset.y() + self.height() / 2
        current_y = 0
        
        # Get page heights from loaded pages
        page_heights = [page_data['height'] for page_data in self.page_pixmaps.values()]
        
        # Find the page that contains the viewport center
        for i, height in enumerate(page_heights):
            page_bottom = current_y + height
            if current_y <= viewport_center < page_bottom:
                if self.current_page != i:
                    self.current_page = i
                    logger.debug(f"Current page updated to {i + 1}")
                break
            current_y = page_bottom

    def resizeEvent(self, event):
        """Handle widget resize"""
        super().resizeEvent(event)
        # Reposition zoom widget
        if hasattr(self, 'zoom_widget'):
            self.zoom_widget.move(self.width() - 90, 10)
            
        # If we have pages, update their display without reloading
        if self.page_pixmaps:
            # Update current page display
            self.update_current_page()
            self.update()
    
    def sizeHint(self):
        """Return the size hint for the widget"""
        if not self.bounding_boxes:
            return super().sizeHint()
            
        # Calculate total height based on loaded pages
        total_height = sum(page_data['height'] for page_data in self.page_pixmaps.values())
        return QSize(self.width(), total_height)
        
    def minimumSizeHint(self):
        """Return the minimum size hint for the widget"""
        if not self.bounding_boxes:
            return super().minimumSizeHint()
            
        # Calculate total height based on loaded pages
        total_height = sum(page_data['height'] for page_data in self.page_pixmaps.values())
        return QSize(self.width(), total_height)
    
    def scroll_to_page(self, page_num):
        """Scroll to make the specified page visible"""
        if not self.bounding_boxes or page_num < 0 or page_num >= len(self.bounding_boxes):
            return
            
        # Calculate the y position for the target page
        y_position = 0
        for i in range(page_num):
            y_position += self.bounding_boxes[i]['structure']['structure']['elements'][0]['attributes']['page_height']
            
        # Center the page in the viewport
        viewport_height = self.height()
        target_y = int(y_position - (viewport_height - self.bounding_boxes[page_num]['structure']['structure']['elements'][0]['attributes']['page_height']) / 2)
        
        # Update pan offset
        self.pan_offset.setY(-target_y)
        self.update()
        logger.debug(f"Scrolled to page {page_num + 1} at y={target_y}")

    def set_current_page(self, page_num):
        """Set the current page and scroll to it"""
        if self.pdf_document and 0 <= page_num < len(self.pdf_document):
            self.current_page = page_num
            self.display_all_pages()
            self.scroll_to_page(page_num)
            logger.debug(f"Set current page to {page_num + 1}")

    def update_current_page_from_scroll(self, scroll_value):
        """Update current page based on scroll position"""
        if not self.bounding_boxes:
            return
            
        # Calculate viewport center
        viewport_height = self.height()
        viewport_center = scroll_value + viewport_height / 2
        
        # Get page heights from loaded pages
        page_heights = [page_data['height'] for page_data in self.page_pixmaps.values()]
        logger.debug(f"Page heights: {page_heights}")
        
        # Find the page that contains the viewport center
        current_height = 0
        for i, height in enumerate(page_heights):
            page_bottom = current_height + height
            logger.debug(f"Page {i} range: {current_height} to {page_bottom}")
            
            if current_height <= viewport_center < page_bottom:
                if self.current_page != i:
                    self.current_page = i
                    logger.debug(f"Current page updated to {i + 1}")
                break
            current_height = page_bottom

    def display_all_pages(self):
        """Display all pages vertically with lazy loading"""
        if not self.pdf_document:
            return
            
        try:
            # Clear existing pixmaps and heights
            self.page_pixmaps = {}
            self.loaded_pages.clear()
            
            # Log initial dimensions
            logger.debug(f"Initial dimensions - width: {self.width()}, height: {self.height()}")
            
            # Load initial set of pages
            for page_num in range(min(self.initial_load_pages, len(self.pdf_document))):
                self.load_page(page_num)
            
            # Calculate total height based on loaded pages
            total_height = sum(page_data['height'] for page_data in self.page_pixmaps.values())
            
            # Set the widget's size to be larger than the viewport
            self.setMinimumHeight(total_height)
            self.setMinimumWidth(self.width())
            
            # Reset pan offset if it would cause empty space at the top
            if self.pan_offset.y() > 0:
                self.pan_offset.setY(0)
            
            # Log final dimensions
            logger.debug(f"Initial display - total height: {total_height}, "
                       f"viewport height: {self.height()}, "
                       f"zoom: {self.zoom}, "
                       f"number of pages loaded: {len(self.page_pixmaps)}")
            
            # Start loading next set of pages in background
            self.load_next_pages()
            
            self.update()
        except Exception as e:
            logger.error(f"Error displaying pages: {str(e)}")
            raise

    def load_next_pages(self):
        """Load next set of pages in background"""
        if self.page_loading or not self.pdf_document:
            return
        
        self.page_loading = True
        try:
            # Find the highest loaded page number
            max_loaded = max(self.loaded_pages) if self.loaded_pages else -1
            
            # Load next set of pages
            next_pages = range(max_loaded + 1, min(max_loaded + self.initial_load_pages + 1, len(self.pdf_document)))
            for page_num in next_pages:
                self.load_page(page_num)
            
            # Update display if current page was loaded
            if self.current_page in self.loaded_pages:
                self.update_current_page()
                self.update()
                
            # Update total height
            total_height = sum(page_data['height'] for page_data in self.page_pixmaps.values())
            self.setMinimumHeight(total_height)
            
            # If there are more pages to load, schedule next batch
            if max_loaded + self.initial_load_pages < len(self.pdf_document):
                QTimer.singleShot(100, self.load_next_pages)
        finally:
            self.page_loading = False

    def load_page(self, page_num):
        """Load a single page and cache its pixmap"""
        if page_num in self.loaded_pages or page_num >= len(self.pdf_document):
            return
        
        try:
            page = self.pdf_document[page_num]
            # Use a higher zoom factor for better quality
            display_zoom = self.zoom
            render_zoom = display_zoom * 2  # Double the zoom for rendering
            matrix = fitz.Matrix(render_zoom, render_zoom)
            pix = page.get_pixmap(matrix=matrix)
            
            # Convert to QImage
            img = QImage(pix.samples, pix.width, pix.height, 
                       pix.stride, QImage.Format.Format_RGB888)
            
            # Create QPixmap
            pixmap = QPixmap.fromImage(img)
            
            # Scale the pixmap to the display size
            display_width = int(self.width() * display_zoom)  # Convert to integer
            # Calculate height maintaining aspect ratio
            aspect_ratio = pixmap.height() / pixmap.width()
            display_height = int(display_width * aspect_ratio)
            
            pixmap = pixmap.scaled(display_width, display_height,
                                 Qt.AspectRatioMode.KeepAspectRatio,
                                 Qt.TransformationMode.SmoothTransformation)
            
            # Store page data
            self.page_pixmaps[page_num] = {
                'pixmap': pixmap,
                'width': display_width,
                'height': display_height
            }
            self.loaded_pages.add(page_num)
            logger.debug(f"Loaded page {page_num + 1} - width: {display_width}, height: {display_height}")
        except Exception as e:
            logger.error(f"Error loading page {page_num}: {str(e)}")

    def leaveEvent(self, event):
        """Handle mouse leave events"""
        if self.hovered_box is not None:
            self.hovered_box = None
            QToolTip.hideText()
            logger.debug("Mouse left widget, hiding tooltip")

    def _check_bounding_box_hover(self, pos):
        """Check if mouse is over a bounding box"""
        # Reset hover state
        self.hovered_box = None
        self.hovered_box_page = None
        
        main_window = self.window()
        if not main_window or not self.bounding_boxes:
            return
            
        # Get scroll position for PDF coordinate calculation only
        scroll_value = main_window.pdf_scroll.verticalScrollBar().value()
        
        # Calculate widget to pixmap ratio for proper scaling
        widget_width = self.width()
        first_page_width = next((self.page_pixmaps[i]['width'] for i in range(len(self.page_pixmaps)) if i in self.page_pixmaps), None)
        if not first_page_width:
            return
            
        # This ratio accounts for how much the PDF is scaled to fit the widget width
        width_ratio = widget_width / (first_page_width * self.zoom)
        
        # Use raw mouse position for viewport coordinates
        viewport_x = pos.x()
        viewport_y = pos.y()
        
        # Calculate absolute y position in document (scaled) - only used for status bar display
        absolute_y = (viewport_y + scroll_value) / self.zoom

        # Find current page based on viewport position
        current_y = 0
        current_page = None

        # Use viewport_y directly for page detection since it's already in screen coordinates
        for page_num in range(len(self.page_pixmaps)):
            if page_num in self.page_pixmaps:
                page_height = self.page_pixmaps[page_num]['height']  # Use actual height, not scaled
                scaled_height = page_height * self.zoom * width_ratio  # Scale height by both zoom and ratio
                if current_y <= viewport_y < current_y + scaled_height:
                    current_page = page_num
                    break
                current_y += scaled_height

        if current_page is None:
            self.dragging_box = None
            self.current_box_page = None
            self.resize_edges = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return

        # Check boxes on current page
        page_boxes = self.bounding_boxes.get(current_page, [])
        for box in page_boxes:
            if 'coordinates' not in box:
                continue
                
            coords = box['coordinates']
            if len(coords) != 4:
                continue
                
            # Get coordinates from the box
            x1 = coords[0]['x']
            y1 = coords[0]['y']
            x2 = coords[2]['x']  # Use point 2 for bottom-right
            y2 = coords[2]['y']
            
            # Get page dimensions for scaling
            page_width = self.page_pixmaps[current_page]['width']
            page_height = self.page_pixmaps[current_page]['height']
            
            # Convert normalized coordinates to screen space, applying both zoom and width ratio
            screen_x1 = x1 * page_width * self.zoom * width_ratio
            screen_x2 = x2 * page_width * self.zoom * width_ratio
            
            # Calculate screen y coordinates relative to viewport, applying both zoom and width ratio
            screen_y1 = y1 * page_height * self.zoom * width_ratio + current_y
            screen_y2 = y2 * page_height * self.zoom * width_ratio + current_y
            
            # Check if mouse is within box boundaries using viewport coordinates
            if (screen_x1 - self.edge_threshold <= viewport_x <= screen_x2 + self.edge_threshold and 
                screen_y1 - self.edge_threshold <= viewport_y <= screen_y2 + self.edge_threshold):
                
                # Determine which edges the cursor is near
                near_left = abs(viewport_x - screen_x1) <= self.edge_threshold
                near_right = abs(viewport_x - screen_x2) <= self.edge_threshold
                near_top = abs(viewport_y - screen_y1) <= self.edge_threshold
                near_bottom = abs(viewport_y - screen_y2) <= self.edge_threshold
                
                self.dragging_box = box
                self.current_box_page = current_page
                self.hovered_box = box
                self.hovered_box_page = current_page
                
                # Set resize edges and cursor based on position
                if near_top and near_left:
                    self.resize_edges = 'nw'
                    self.setCursor(Qt.CursorShape.SizeFDiagCursor)
                elif near_top and near_right:
                    self.resize_edges = 'ne'
                    self.setCursor(Qt.CursorShape.SizeBDiagCursor)
                elif near_bottom and near_left:
                    self.resize_edges = 'sw'
                    self.setCursor(Qt.CursorShape.SizeBDiagCursor)
                elif near_bottom and near_right:
                    self.resize_edges = 'se'
                    self.setCursor(Qt.CursorShape.SizeFDiagCursor)
                elif near_left:
                    self.resize_edges = 'w'
                    self.setCursor(Qt.CursorShape.SizeHorCursor)
                elif near_right:
                    self.resize_edges = 'e'
                    self.setCursor(Qt.CursorShape.SizeHorCursor)
                elif near_top:
                    self.resize_edges = 'n'
                    self.setCursor(Qt.CursorShape.SizeVerCursor)
                elif near_bottom:
                    self.resize_edges = 's'
                    self.setCursor(Qt.CursorShape.SizeVerCursor)
                else:
                    self.resize_edges = None
                    self.setCursor(Qt.CursorShape.OpenHandCursor if not self.drag_start_pos else Qt.CursorShape.ClosedHandCursor)
                
                # Show box information in status bar
                category = box.get('category', 'Unknown').capitalize()
                # Convert viewport coordinates to PDF coordinates for display
                pdf_x = viewport_x / (self.zoom * width_ratio)
                pdf_y = absolute_y
                status_msg = (f"Page {current_page + 1} | "
                            f"Category: {category} | "
                            f"Box: ({x1:.2f}, {y1:.2f}) to ({x2:.2f}, {y2:.2f}) | "
                            f"Viewport: ({viewport_x}, {viewport_y}) | "
                            f"PDF: ({pdf_x:.1f}, {pdf_y:.1f}) | "
                            f"Zoom: {self.zoom:.1f}x | Ratio: {width_ratio:.2f}")
                self.current_cursor_page = current_page
                main_window.status_label.showMessage(status_msg, 0)
                return

        # If we get here, mouse is not over any box
        self.dragging_box = None
        self.current_box_page = None
        self.resize_edges = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        # Reset status bar to default coordinate display
        # Convert viewport coordinates to PDF coordinates for display
        pdf_x = viewport_x / (self.zoom * width_ratio)
        pdf_y = absolute_y
        if hasattr(main_window, 'status_label'):
            main_window.status_label.showMessage(
                f"Page {current_page + 1} | Viewport: ({viewport_x}, {viewport_y}) | PDF: ({pdf_x:.1f}, {pdf_y:.1f}) | Zoom: {self.zoom:.1f}x | Ratio: {width_ratio:.2f}", 
                0
            )

    def start_element_creation(self):
        """Start the element creation process"""
        self.creating_element = True
        self.setCursor(Qt.CursorShape.CrossCursor)
        # Show element type selection dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Element Type")
        layout = QVBoxLayout()
        
        # Create radio buttons for element types
        button_group = QButtonGroup()
        types = ['text', 'figure', 'table', 'picture', 'caption']
        for element_type in types:
            radio = QRadioButton(element_type.capitalize())
            radio.setChecked(element_type == 'text')
            button_group.addButton(radio)
            layout.addWidget(radio)
        
        # Add OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Get selected type
            for button in button_group.buttons():
                if button.isChecked():
                    self.element_type = button.text().lower()
                    break
        else:
            self.creating_element = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

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

        caption_text = QTextEdit()
        caption_text.setReadOnly(True)
        caption_text.setPlainText(self.element_data.get('caption', ''))
        caption_text.setMaximumHeight(100)  # Limit height for captions
        caption_text.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout.addWidget(caption_text)
        
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
        """Initialize the UI"""
        # Create main layout
        layout = QVBoxLayout()
        
        # Create toolbar for view mode switching
        toolbar = QToolBar()
        toolbar.setMovable(False)
        
        # Create view mode actions
        self.list_view_action = QAction("List View", self)
        self.list_view_action.setCheckable(True)
        self.list_view_action.setChecked(False)
        self.list_view_action.triggered.connect(lambda: self.switch_view_mode('list'))
        
        self.icon_view_action = QAction("Icon View", self)
        self.icon_view_action.setCheckable(True)
        self.icon_view_action.setChecked(True)  # Set icon view as default
        self.icon_view_action.triggered.connect(lambda: self.switch_view_mode('icon'))
        
        # Add actions to toolbar
        toolbar.addAction(self.list_view_action)
        toolbar.addAction(self.icon_view_action)
        
        # Add Save to Database button
        self.save_action = QAction("Save to Database", self)
        self.save_action.setEnabled(False)  # Disabled for now until we implement the feature
        self.save_action.setToolTip("This feature will be implemented in a future update")
        toolbar.addAction(self.save_action)
        
        layout.addWidget(toolbar)
        
        # Create list widget for both views
        self.content_list = QListWidget()
        self.content_list.setViewMode(QListWidget.ViewMode.IconMode)  # Set icon mode as default
        self.content_list.setIconSize(QSize(200, 200))
        self.content_list.setSpacing(10)
        self.content_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.content_list.setWrapping(False)  # Single column
        self.content_list.setFlow(QListWidget.Flow.TopToBottom)  # Vertical flow
        self.content_list.itemClicked.connect(self._handle_item_click)
        self.content_list.itemDoubleClicked.connect(self._show_element_info)
        
        layout.addWidget(self.content_list)
        
        self.setLayout(layout)
        
        # Initialize with icon view
        self.current_view_mode = 'icon'
        self.switch_view_mode('icon')
        
    def _handle_item_click(self, item):
        """Handle single click on an item to scroll to the corresponding page"""
        # Get the main window instance
        main_window = self.window()
        if not main_window:
            return
            
        # Get the widget associated with the item
        widget = self.content_list.itemWidget(item)
        if not widget or not hasattr(widget, 'element_data'):
            return
            
        # Get the page number from the element data
        page_num = widget.element_data.get('page')
        logger.debug(f"Page number: {page_num}")
        if page_num is not None:
            # Update current page in PDF viewer
            main_window.current_page = page_num - 1  # Convert to 0-based index
            main_window.pdf_viewer.current_page = page_num - 1
            main_window.pdf_viewer.display_all_pages()
            
            # Calculate the scroll position for the target page
            scroll_bar = main_window.pdf_scroll.verticalScrollBar()
            target_pos = 0
            
            # Sum up the heights of all pages before the target page
            for i in range(page_num - 1):
                if i in main_window.pdf_viewer.page_pixmaps:
                    target_pos += main_window.pdf_viewer.page_pixmaps[i]['height']
            
            # Scroll to the target position
            scroll_bar.setValue(target_pos)
            main_window.update_navigation()
        
    def set_document(self, doc):
        """Set the current PDF document"""
        self.current_doc = doc
        
    def switch_view_mode(self, mode):
        """Switch between list and icon views"""
        if mode == 'list':
            self.content_list.setViewMode(QListWidget.ViewMode.ListMode)
            self.content_list.setIconSize(QSize(32, 32))  # Smaller icons for list view
            self.content_list.setSpacing(2)  # Less spacing for list view
            self.list_view_action.setChecked(True)
            self.icon_view_action.setChecked(False)
        else:
            self.content_list.setViewMode(QListWidget.ViewMode.IconMode)
            self.content_list.setIconSize(QSize(200, 200))  # Larger icons for icon view
            self.content_list.setSpacing(10)  # More spacing for icon view
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
        logger.debug("StructuredContentView.update_content called")
        logger.debug(f"Received page_structures with {len(page_structures)} pages")
        
        self._current_content = page_structures  # Store for view switching
        
        # Clear current view
        self.content_list.clear()
        
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
            logger.debug(f"Processing page {page_num} with {len(elements)} elements")
            for element in elements:
                element_type = element.get('category', '').lower()
                if element_type in content_by_type:
                    # Find nearest caption
                    caption_element = self._find_nearest_caption(element, elements)
                    logger.debug(f"Caption element: {caption_element}")
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
                    logger.debug(f"Added element of type {element_type} from page {page_num}")
        
        # Sort elements by position for each type
        for content_type in content_by_type:
            content_by_type[content_type] = self._sort_elements_by_position(content_by_type[content_type])
            logger.debug(f"Sorted {len(content_by_type[content_type])} {content_type} elements")
        
        # Update view based on current mode
        if self.content_list.viewMode() == QListWidget.ViewMode.ListMode:
            logger.debug("Updating list view")
            self._update_list_view(content_by_type)
        else:
            logger.debug("Updating icon view")
            self._update_icon_view(content_by_type)
    
    def _update_list_view(self, content_by_type):
        """Update the list view with content"""
        for content_type, items in content_by_type.items():
            if items:
                # Add type label as a special item
                type_item = QListWidgetItem(content_type.capitalize())
                type_item.setFlags(Qt.ItemFlag.NoItemFlags)  # Make it non-selectable
                type_item.setForeground(Qt.GlobalColor.black)
                font = type_item.font()
                font.setBold(True)
                type_item.setFont(font)
                self.content_list.addItem(type_item)
                
                for item in items:
                    # Create item widget
                    item_widget = QWidget()
                    item_layout = QGridLayout()  # Use grid layout for fixed columns
                    item_layout.setContentsMargins(5, 5, 5, 5)
                    item_layout.setSpacing(10)  # Add some spacing between columns
                    
                    # Add page number (column 0)
                    page_label = QLabel(f"Page {item['page']}")
                    page_label.setFixedWidth(80)  # Fixed width for page number
                    item_layout.addWidget(page_label, 0, 0)
                    
                    # Add content (column 1)
                    content_label = QLabel(item['content'])
                    content_label.setWordWrap(True)
                    content_label.setFixedWidth(300)  # Fixed width for content
                    item_layout.addWidget(content_label, 0, 1)
                    
                    # Add caption if available (column 2)
                    if item['caption']:
                        caption_label = QLabel(item['caption'])
                        caption_label.setWordWrap(True)
                        caption_label.setStyleSheet("font-style: italic;")
                        caption_label.setFixedWidth(300)  # Fixed width for caption
                        item_layout.addWidget(caption_label, 0, 2)
                    
                    item_widget.setLayout(item_layout)
                    
                    # Store element data for double-click
                    item_widget.element_data = {
                        'type': content_type,
                        'page': item['page'],
                        'caption': item['caption'],
                        'content': item['content'],
                        'pixmap': None  # No pixmap for list view
                    }
                    
                    # Create list item
                    list_item = QListWidgetItem()
                    list_item.setSizeHint(item_widget.sizeHint())
                    self.content_list.addItem(list_item)
                    self.content_list.setItemWidget(list_item, item_widget)
                
                # Add spacing after each type
                spacer = QListWidgetItem()
                spacer.setSizeHint(QSize(0, 10))
                spacer.setFlags(Qt.ItemFlag.NoItemFlags)
                self.content_list.addItem(spacer)
    
    def _update_icon_view(self, content_by_type):
        """Update the icon view with content"""
        self.content_list.clear()
        
        if not self.current_doc:
            logger.warning("No PDF document available for icon view")
            return
            
        # Flatten all items into a single list
        all_items = []
        for items in content_by_type.values():
            all_items.extend(items)
            
        # Sort items by page number and position
        all_items = self._sort_elements_by_position(all_items)
        
        # Add items in chronological order
        for item in all_items:
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
                logger.error(f"Error creating image for item on page {item['page']}: {str(e)}")
                item_pixmap = None
            
            # Add page number and type label
            header_label = QLabel(f"Page {item['page']} - {item.get('type', 'Unknown')}")
            header_label.setStyleSheet("font-weight: bold;")
            item_layout.addWidget(header_label)
            
            # Show caption if available
            if item['caption']:
                caption_label = QLabel(item['caption'])
                caption_label.setWordWrap(True)
                caption_label.setStyleSheet("font-style: italic;")
                item_layout.addWidget(caption_label)
            
            item_widget.setLayout(item_layout)
            item_widget.setStyleSheet("border: 1px solid #ccc; padding: 5px;")
            
            # Store element data for double-click
            item_widget.element_data = {
                'type': item.get('type', 'Unknown'),
                'page': item['page'],
                'caption': item['caption'],
                'content': item['content'],
                'pixmap': item_pixmap
            }
            
            # Create list item
            list_item = QListWidgetItem()
            list_item.setSizeHint(item_widget.sizeHint())
            self.content_list.addItem(list_item)
            self.content_list.setItemWidget(list_item, item_widget)
            
            # Add spacing between items
            spacer = QListWidgetItem()
            spacer.setSizeHint(QSize(0, 10))
            spacer.setFlags(Qt.ItemFlag.NoItemFlags)
            self.content_list.addItem(spacer)
    
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

    def _show_element_info(self, item):
        """Show element information dialog"""
        # Get the main window instance
        main_window = self.window()
        if not main_window:
            return
            
        # Get the widget associated with the item
        widget = self.content_list.itemWidget(item)
        if not widget or not hasattr(widget, 'element_data'):
            return
            
        # Get the page number from the element data
        page_num = widget.element_data.get('page')
        logger.debug(f"Page number: {page_num}")
        if page_num is not None:
            # Update current page in PDF viewer
            main_window.current_page = page_num - 1  # Convert to 0-based index
            main_window.pdf_viewer.current_page = page_num - 1
            main_window.pdf_viewer.display_all_pages()
            
            # Calculate the scroll position for the target page
            scroll_bar = main_window.pdf_scroll.verticalScrollBar()
            target_pos = 0
            
            # Sum up the heights of all pages before the target page
            for i in range(page_num - 1):
                if i in main_window.pdf_viewer.page_pixmaps:
                    target_pos += main_window.pdf_viewer.page_pixmaps[i]['height']
            
            # Scroll to the target position
            scroll_bar.setValue(target_pos)
            main_window.update_navigation()
            
        # Show element info dialog
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
            'initial_page_structures': {},
            'metadata': {}
        }
        self.recent_files = []

        # initialize variables
        self.current_file_path = None
        self.current_file_directory = None
        self.pdf_document= None
        self.current_page = 0

        # Create status bar
        self.status_label = self.statusBar()
        self.status_label.showMessage("Ready")
        
        # Create directory tree widget
        self.create_directory_tree()
        
        # Create main splitter for left and right panels
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(main_splitter)
        
        # Add library dock to main splitter
        main_splitter.addWidget(self.library_dock)
        
        # Create right panel splitter
        right_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(right_splitter)
        
        # Create PDF viewer with scroll area
        self.pdf_scroll = QScrollArea()
        self.pdf_scroll.setWidgetResizable(True)
        self.pdf_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.pdf_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.pdf_scroll.setFrameShape(QFrame.Shape.NoFrame)  # Remove frame
        self.pdf_scroll.setViewportMargins(0, 0, 0, 0)  # Remove margins
        
        # Connect scroll events
        self.pdf_scroll.verticalScrollBar().valueChanged.connect(self.handle_scroll)
        
        # Create container widget for PDF viewer
        pdf_container = QWidget()
        pdf_layout = QVBoxLayout(pdf_container)
        pdf_layout.setContentsMargins(0, 0, 0, 0)
        pdf_layout.setSpacing(0)
        
        self.pdf_viewer = PDFViewer()
        pdf_layout.addWidget(self.pdf_viewer)
        
        self.pdf_scroll.setWidget(pdf_container)
        right_splitter.addWidget(self.pdf_scroll)
        
        # Create structured content view with scroll area
        content_scroll = QScrollArea()
        content_scroll.setWidgetResizable(True)
        self.structured_view = StructuredContentView()
        content_scroll.setWidget(self.structured_view)
        right_splitter.addWidget(content_scroll)
        
        # Set initial splitter sizes (40% for library, 30% for PDF, 30% for content)
        main_splitter.setSizes([400, 800])
        right_splitter.setSizes([400, 400])
        
        self.create_toolbar()
        self.create_menu()
        logger.info(f"{PROGRAM_NAME} v{PROGRAM_VERSION} started")

        # Add recent files list
        self.load_recent_files()

        # Initialize collection items cache
        self.collection_items_cache = {}

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
                'initial_page_structures': {},
                'metadata': {}
            }
            
            # Load items for this collection
            self.load_collection_items(item.collection_id)
            
            logger.debug(f"Showing items for collection: {item.text(0)}")

    def item_clicked(self, item, column):
        """Handle click on item in items tree"""
        try:
            # Set wait cursor
            QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
            
            # Clear existing document data and session information
            self.document_data = {
                'page_structures': {},
                'initial_page_structures': {},
                'metadata': {}
            }
            self.pdf_viewer.set_bounding_boxes([])
            self.pdf_viewer.pixmap = None
            self.pdf_viewer.update()
            
            # Scroll PDF view area to top
            self.pdf_scroll.verticalScrollBar().setValue(0)
            
            if hasattr(item, 'file_path') and item.file_path:
                # Direct click on a PDF item
                if item.file_path.lower().endswith('.pdf'):
                    # Clear bounding boxes before loading new PDF
                    self.pdf_viewer.bounding_boxes.clear()
                    self.pdf_viewer.set_bounding_boxes([])
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
                    # Clear bounding boxes before loading new PDF
                    self.pdf_viewer.bounding_boxes.clear()
                    self.pdf_viewer.set_bounding_boxes([])
                    self.load_pdf_file(pdf_item.file_path)
                    logger.debug(f"Loading single PDF attachment: {pdf_item.file_path}")
                elif len(pdf_items) > 1:
                    # If there are multiple PDFs, load the first one and log a message
                    pdf_item = pdf_items[0]
                    # Clear bounding boxes before loading new PDF
                    self.pdf_viewer.bounding_boxes.clear()
                    self.pdf_viewer.set_bounding_boxes([])
                    self.load_pdf_file(pdf_item.file_path)
                    logger.info(f"Loading first of {len(pdf_items)} PDF attachments: {pdf_item.file_path}")
        finally:
            # Restore cursor
            QApplication.restoreOverrideCursor()
            self.ensure_normal_cursor()

    def load_pdf_file(self, file_path):
        """Load a PDF file and try to load its analysis from database"""
        self.current_file_path = file_path
        self.current_file_directory = os.path.dirname(file_path)
        self.document_data = {
            'page_structures': {},
            'initial_page_structures': {},
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
        
        self.pdf_document= fitz.open(file_path)
        self.current_page = 0
        self.pdf_viewer.current_page = 0
        self.pdf_viewer.open_pdf(file_path)
        self.update_navigation()
        self.status_label.showMessage(f"Opened: {os.path.basename(file_path)}", 3000)
        logger.info(f"Opened PDF file: {self.current_file_path}")
        
        # Set document in structured view
        self.structured_view.set_document(self.pdf_document)
        
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

    def load_document_from_database(self, file_path, zotero_key):
        """Load a PDF file and try to load its analysis from database"""
        self.current_file_path = file_path
        self.current_file_directory = os.path.dirname(file_path)
        self.pdf_document= fitz.open(file_path)
        self.current_page = 0
        self.pdf_viewer.current_page = 0
        

    def set_bounding_boxes(self, boxes):
        """Set bounding boxes for all pages"""
        if not boxes:
            return
            
        # Clear existing boxes
        self.bounding_boxes.clear()
        
        # Get the main window instance
        main_window = self.window()
        if not hasattr(main_window, 'document_data'):
            return
            
        # Get all page structures
        page_structures = main_window.document_data.get('page_structures', {})
        
        # Set boxes for each page
        for page_num, structure in page_structures.items():
            page_boxes = structure.get('structure', {}).get('elements', [])
            if page_boxes:
                self.bounding_boxes.append({
                    'page': int(page_num) + 1,
                    'structure': structure,
                    'coordinates': page_boxes
                })
                logger.debug(f"Set {len(page_boxes)} bounding boxes for page {int(page_num) + 1}")
        
        self.update()

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
                self.current_file_path_index = 0
                
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
        """Create the main toolbar"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
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
        analyze_action.triggered.connect(lambda: self.analyze_pdf(self.current_file_path) if self.current_file_path else None)
        toolbar.addAction(analyze_action)
        
        # Add separator
        toolbar.addSeparator()
        
        # Batch analyze action
        batch_analyze_action = QAction("Batch Analyze", self)
        batch_analyze_action.triggered.connect(self.batch_analyze)
        toolbar.addAction(batch_analyze_action)
        
        # Add separator
        toolbar.addSeparator()
        
        # Add Element action
        add_element_action = QAction("Add Element", self)
        add_element_action.setShortcut("Ctrl+N")
        add_element_action.triggered.connect(self.pdf_viewer.start_element_creation)
        toolbar.addAction(add_element_action)

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
            self.current_file_path = file_path
            self.current_file_directory = os.path.dirname(file_path)
            self.document_data = {
                'page_structures': {},
                'initial_page_structures': {},
                'metadata': {}
            }
            self.pdf_document= fitz.open(file_path)
            self.current_page = 0
            self.pdf_viewer.current_page = 0
            self.pdf_viewer.open_pdf(file_path)
            self.update_navigation()
            self.status_label.showMessage(f"Opened: {os.path.basename(file_path)}", 3000)
            logger.info(f"Opened PDF file: {self.current_file_path}")
            
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
                sample_size = max(1, len(self.pdf_document) // 10)  # 10% of pages or at least 1 page
                logger.info(f"Checking first {sample_size} pages for text content using PyMuPDF")
                
                # Check text content in sample pages using PyMuPDF
                page_char_counts = {}
                for page_num in range(sample_size):
                    try:
                        page = self.pdf_document[page_num]
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
                                self.document_data['initial_page_structures'][page_num] = {
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

    def load_document_from_database(self, file_path, zotero_key=None, file_hash=None):
        """Load document from database for a given file"""
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
                        
                if not document:
                    try:
                        document = PDFDocument.get(PDFDocument.file_path == file_path)
                        logger.debug(f"Found document by file path: {file_path}")
                    except DoesNotExist:
                        logger.debug(f"No document found with file path: {file_path}")
                        return False
                
            return document
        except Exception as e:
            logger.error(f"Error loading document from database: {str(e)}")
            return False

    def get_zotero_key_for_current_file(self):
        """Get Zotero key for the current file"""
        try:
            current_item = self.items_tree.currentItem()
            if current_item and hasattr(current_item, 'zotero_key'):
                return current_item.zotero_key
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting Zotero key for current file: {str(e)}")
            return None

    def save_session(self):
        """Save current session data to database"""
        try:
            if not hasattr(self, 'current_file_path') or not self.current_file_path:
                logger.debug("No current file to save session for")
                return

            file_hash = None
            zotero_key = None
            zotero_key = self.get_zotero_key_for_current_file()
            if not zotero_key:
                logger.debug("No Zotero key found for current file, using file hash")
                file_hash = calculate_file_hash(self.current_file_path)
            else:
                file_hash = None

            document = self.load_document_from_database(self.current_file_path, zotero_key, file_hash)

            with db:
                # Update document fields if needed
                if document:
                    if not document.file_hash and file_hash:
                        document.file_hash = file_hash
                    if not document.zotero_key and zotero_key:
                        document.zotero_key = zotero_key
                    if not document.page_count and self.pdf_document:
                        document.page_count = len(self.pdf_document)
                    document.save()
                    logger.debug(f"Updated document record")
                
                # Save structured elements
                if 'page_structures' in self.document_data:
                    logger.debug("Saving structured elements to database")
                    # Delete existing elements for this document
                    StructuredElement.delete().where(StructuredElement.document == document).execute()
                    
                    # Create new elements
                    elements_to_create = []
                    for page_num, page_data in self.document_data['page_structures'].items():
                        if 'structure' in page_data and 'elements' in page_data['structure']:
                            for element in page_data['structure']['elements']:
                                logger.debug(f"Creating element: {element}")
                                elements_to_create.append({
                                    'document': document,
                                    'page_number': int(page_num),
                                    'element_id': str(element.get('id', '')),
                                    'element_type': element.get('category', 'unknown'),
                                    'coordinates': json.dumps(element.get('coordinates', [])),
                                    'content': json.dumps(element.get('content', {})),
                                    'caption': json.dumps(element.get('caption', {})),
                                    'metadata': json.dumps(element.get('metadata', {})),
                                    'created_at': datetime.datetime.now(),
                                    'updated_at': datetime.datetime.now()
                                })
                    # printout elements' page and id, not the whole element for elements_to_create
                    logger.info(f"Elements to create: {[{'page': e['page_number'], 'id': e['element_id']} for e in elements_to_create]}")

                    if elements_to_create:
                        with db.atomic():
                            for batch in chunked(elements_to_create, 100):
                                StructuredElement.insert_many(batch).execute()
                        logger.debug(f"Saved {len(elements_to_create)} structured elements")
                
                # Prepare session data
                session_data = {
                    'document_data': {
                        'page_structures': self.document_data['page_structures'],
                        'initial_page_structures': self.document_data['initial_page_structures'],
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
                
                logger.info(f"Session saved to database for {self.current_file_path} (Zotero key: {zotero_key})")
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

            zotero_key = None
            file_hash = None
            zotero_key = self.get_zotero_key_for_current_file()
            if not zotero_key:
                logger.debug("No Zotero key found for current file, using file hash")
                file_hash = calculate_file_hash(self.current_file_path)
            else:
                file_hash = None
            
            document = self.load_document_from_database(file_path, zotero_key, file_hash)
            if not document:
                logger.error(f"No document found in database for {file_path}")
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
            
            # Load and merge structured elements
            elements = (StructuredElement
                      .select()
                      .where(StructuredElement.document == document)
                      .order_by(StructuredElement.page_number))
            
            logger.debug(f"Loading {elements.count()} structured elements")
            for element in elements:
                page_num = str(element.page_number)
                if page_num not in page_structures:
                    page_structures[page_num] = {'structure': {'elements': []}}
                
                # Convert element to dictionary format
                coords = json.loads(element.coordinates)
                if len(coords) >= 4:
                    # Normalize coordinates to ensure x1 < x2 and y1 < y2
                    x1 = min(coords[0]['x'], coords[2]['x'])
                    y1 = min(coords[0]['y'], coords[2]['y'])
                    x2 = max(coords[0]['x'], coords[2]['x'])
                    y2 = max(coords[0]['y'], coords[2]['y'])
                    
                    normalized_coords = [
                        {'x': x1, 'y': y1},  # top-left
                        {'x': x2, 'y': y1},  # top-right
                        {'x': x2, 'y': y2},  # bottom-right
                        {'x': x1, 'y': y2}   # bottom-left
                    ]
                    
                    element_data = {
                        'id': element.element_id,
                        'category': element.element_type,
                        'coordinates': normalized_coords,
                        'content': json.loads(element.content) if element.content else {},
                        'caption': json.loads(element.caption) if element.caption else {},
                        'metadata': json.loads(element.metadata) if element.metadata else {}
                    }
                    
                    # Update existing element or append new one
                    elements_list = page_structures[page_num]['structure']['elements']
                    for i, existing in enumerate(elements_list):
                        if str(existing.get('id')) == str(element.element_id):
                            elements_list[i] = element_data
                            break
                    else:
                        elements_list.append(element_data)
            
            # Open the PDF file
            try:
                self.current_file_path = file_path
                self.current_file_directory = os.path.dirname(file_path)
                self.pdf_document= fitz.open(file_path)
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
                
                logger.info(f"Successfully loaded session with {len(self.pdf_document)} pages (Zotero key: {zotero_key})")
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
            if self.pdf_document is None:
                return
            
            # Update PDF display
            self.pdf_viewer.current_page = self.current_page
            self.pdf_viewer.display_all_pages()

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
            self.total_pages_label.setText(f"/ {len(self.pdf_document)}")
            
        except Exception as e:
            logger.error(f"Error updating page display: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            self.status_label.showMessage(f"Error updating display: {str(e)}", 3000)

    def closeEvent(self, event):
        # Save window geometry before closing
        self.settings.setValue("window_geometry", self.saveGeometry())
        super().closeEvent(event)

    def prev_page(self):
        if self.pdf_document and self.current_page > 0:
            self.current_page -= 1
            self.pdf_viewer.current_page = self.current_page
            self.pdf_viewer.display_all_pages()
            
            # Get the current scroll position
            scroll_bar = self.pdf_scroll.verticalScrollBar()
            current_pos = scroll_bar.value()
            
            # Calculate the height of one page
            page_height = self.pdf_viewer.page_pixmaps[self.current_page]['height']
            
            # Scroll up by one page height
            scroll_bar.setValue(current_pos - page_height)
            self.update_navigation()
            logger.debug(f"Navigated to previous page: {self.current_page + 1}")
            
    def next_page(self):
        if self.pdf_document and self.current_page < len(self.pdf_document) - 1:
            self.current_page += 1
            self.pdf_viewer.current_page = self.current_page
            self.pdf_viewer.display_all_pages()
            
            # Get the current scroll position
            scroll_bar = self.pdf_scroll.verticalScrollBar()
            current_pos = scroll_bar.value()
            
            # Calculate the height of one page
            page_height = self.pdf_viewer.page_pixmaps[self.current_page]['height']
            
            # Scroll down by one page height
            scroll_bar.setValue(current_pos + page_height)
            self.update_navigation()
            logger.debug(f"Navigated to next page: {self.current_page + 1}")
            
    def update_navigation(self):
        """Update navigation controls and page display"""
        if self.pdf_document:
            self.current_page_input.setText(str(self.current_page + 1))
            self.total_pages_label.setText(f"/ {len(self.pdf_document)}")
            self.prev_button.setEnabled(self.current_page > 0)
            self.next_button.setEnabled(self.current_page < len(self.pdf_document) - 1)
            
            # Update bounding boxes for current page
            if self.pdf_viewer.show_bounding_boxes:
                current_page_boxes = self.document_data['page_structures'].get(str(self.current_page), {})
                if current_page_boxes:
                    self.pdf_viewer.set_bounding_boxes(current_page_boxes.get('structure', {}).get('elements', []))
                else:
                    self.pdf_viewer.set_bounding_boxes([])

    def go_to_page(self):
        if not self.pdf_document:
            return
            
        try:
            page_num = int(self.current_page_input.text()) - 1  # Convert to 0-based index
            if 0 <= page_num < len(self.pdf_document):
                self.current_page = page_num
                self.pdf_viewer.current_page = self.current_page
                self.pdf_viewer.display_all_pages()
                
                # Calculate the scroll position for the target page
                scroll_bar = self.pdf_scroll.verticalScrollBar()
                target_pos = 0
                
                # Sum up the heights of all pages before the target page
                for i in range(page_num):
                    if i in self.pdf_viewer.page_pixmaps:
                        target_pos += self.pdf_viewer.page_pixmaps[i]['height']
                
                # Scroll to the target position
                scroll_bar.setValue(target_pos)
                self.update_navigation()
                logger.debug(f"Navigated to page: {self.current_page + 1}")
            else:
                QMessageBox.warning(self, "Invalid Page", 
                                  f"Page number must be between 1 and {len(self.pdf_document)}")
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
            
            # Store the database path and directory for later use
            self.zotero_db_path = db_path
            self.zotero_dir = zotero_dir
            
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
            
            # Clear existing items
            self.collections_tree.clear()
            self.items_tree.clear()
            
            # Create collection hierarchy first
            collection_map = {}
            root_collections = []
            self.collection_data = {}
            
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
            
            # Add collections to the collections tree immediately
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

    def load_collection_items(self, collection_id):
        """Load items for a specific collection"""
        try:
            # Check if items are already in cache
            if collection_id in self.collection_items_cache:
                logger.debug(f"Retrieving items for collection {collection_id} from cache")
                self._display_cached_collection_items(collection_id)
                return
            
            # Set wait cursor
            QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
            self.status_label.showMessage("Loading collection items...", 0)
            QApplication.processEvents()
            
            # Connect to the database in read-only mode
            uri = f"file:{self.zotero_db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
            cursor = conn.cursor()
            
            # Get parent items and their collections
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
                WHERE ci.collectionID = ?
                AND i.itemID NOT IN (SELECT itemID FROM itemAttachments)
                AND it.typeName NOT IN ('attachment', 'note')
                AND (id.fieldID IS NULL OR id.fieldID = (
                    SELECT fieldID FROM fields WHERE fieldName = 'title'
                ))
                ORDER BY idv.value
            """, (collection_id,))
            
            parent_items = cursor.fetchall()
            logger.debug(f"Found {len(parent_items)} parent items for collection {collection_id}")
            
            # Get all PDF attachments for these items
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
                AND (a.parentItemID IN (SELECT itemID FROM collectionItems WHERE collectionID = ?)
                     OR ci.collectionID = ?)
            """, (collection_id, collection_id))
            
            attachments = cursor.fetchall()
            logger.info(f"Found {len(attachments)} PDF attachments for collection {collection_id}")
            
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
            
            # Process items and attachments
            storage_base = os.path.join(self.zotero_dir, 'storage')
            
            # Function to add PDF item to items tree and save to database
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
                        pdf_item.zotero_key = zotero_key
                        if parent_key:
                            pdf_item.parent_zotero_key = parent_key
                        
                        # Save document to database only if it doesn't exist
                        try:
                            with db:
                                try:
                                    document = PDFDocument.get(PDFDocument.zotero_key == zotero_key)
                                    logger.debug(f"Document already exists in database: {zotero_key}")
                                except DoesNotExist:
                                    document = PDFDocument.create(
                                        file_path=pdf_path,
                                        zotero_key=zotero_key,
                                        title=display_name,
                                        page_count=0
                                    )
                                    logger.debug(f"Created new document record for {display_name}")
                        except Exception as e:
                            logger.error(f"Error checking/saving document to database: {str(e)}")
                        
                        return True
                    else:
                        logger.warning(f"No PDF files found in directory: {storage_dir}")
                else:
                    logger.warning(f"Storage directory not found: {storage_dir}")
                return False
            
            # Clear existing items
            self.items_tree.clear()
            
            # Store items in cache
            collection_items = []
            
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
                    
                    # Create new parent item
                    parent_item = QTreeWidgetItem()
                    parent_item.setText(0, display_text)
                    parent_item.item_id = item_id
                    parent_item.zotero_key = key
                    self.items_tree.addTopLevelItem(parent_item)
                    
                    # Store parent item in cache
                    parent_item_data = {
                        'text': display_text,
                        'item_id': item_id,
                        'zotero_key': key,
                        'children': []
                    }
                    
                    # Add PDF attachments as children
                    for attachment in attachments_by_parent[item_id]:
                        try:
                            zotero_key = attachment[4]  # Zotero key
                            parent_key = attachment[6]  # Parent's Zotero key
                            display_name = attachment[3]
                            if display_name.startswith('storage:'):
                                display_name = os.path.basename(display_name)
                            
                            if add_pdf_to_items_tree(zotero_key, display_name, parent_item, collection_id, parent_key):
                                # Store child item in cache
                                child_item_data = {
                                    'text': display_name,
                                    'zotero_key': zotero_key,
                                    'parent_zotero_key': parent_key,
                                    'file_path': os.path.join(storage_base, zotero_key, [f for f in os.listdir(os.path.join(storage_base, zotero_key)) if f.lower().endswith('.pdf')][0])
                                }
                                parent_item_data['children'].append(child_item_data)
                                
                        except Exception as e:
                            logger.error(f"Error processing attachment {zotero_key}: {str(e)}")
                    
                    collection_items.append(parent_item_data)
            
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
                    
                    if add_pdf_to_items_tree(zotero_key, display_name, None, attachment[5]):  # attachment[5] is collectionID
                        # Store standalone item in cache
                        standalone_item_data = {
                            'text': display_name,
                            'zotero_key': zotero_key,
                            'file_path': os.path.join(storage_base, zotero_key, [f for f in os.listdir(os.path.join(storage_base, zotero_key)) if f.lower().endswith('.pdf')][0])
                        }
                        collection_items.append(standalone_item_data)
                    
                except Exception as e:
                    logger.error(f"Error processing standalone attachment {zotero_key}: {str(e)}")
            
            # Store items in cache
            self.collection_items_cache[collection_id] = collection_items
            
            self.status_label.showMessage("Loaded collection items", 3000)
            logger.info(f"Loaded items for collection {collection_id}")
            
        except Exception as e:
            logger.error(f"Error loading collection items: {str(e)}")
            QMessageBox.warning(self, "Error", f"Could not load collection items:\n{str(e)}")
            
        finally:
            # Restore cursor
            QApplication.restoreOverrideCursor()
            self.ensure_normal_cursor()

    def _display_cached_collection_items(self, collection_id):
        """Display items from cache for a collection"""
        try:
            # Clear existing items
            self.items_tree.clear()
            
            # Get items from cache
            collection_items = self.collection_items_cache[collection_id]
            
            # Display items
            for item_data in collection_items:
                # Create parent item
                parent_item = QTreeWidgetItem()
                parent_item.setText(0, item_data['text'])
                if 'item_id' in item_data:
                    parent_item.item_id = item_data['item_id']
                if 'zotero_key' in item_data:
                    parent_item.zotero_key = item_data['zotero_key']
                self.items_tree.addTopLevelItem(parent_item)
                
                # Add children if any
                for child_data in item_data.get('children', []):
                    child_item = QTreeWidgetItem(parent_item)
                    child_item.setText(0, child_data['text'])
                    child_item.zotero_key = child_data['zotero_key']
                    if 'parent_zotero_key' in child_data:
                        child_item.parent_zotero_key = child_data['parent_zotero_key']
                    if 'file_path' in child_data:
                        child_item.file_path = child_data['file_path']
            
            self.status_label.showMessage("Loaded collection items from cache", 3000)
            logger.debug(f"Displayed cached items for collection {collection_id}")
            
        except Exception as e:
            logger.error(f"Error displaying cached collection items: {str(e)}")
            QMessageBox.warning(self, "Error", f"Could not display cached collection items:\n{str(e)}")

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

            # Get currently selected collection from main window
            current_collection = self.collections_tree.currentItem()
            if current_collection and hasattr(current_collection, 'collection_id'):
                # Find and select the same collection in the dialog's tree
                def find_and_select_collection(item, target_id):
                    if hasattr(item, 'collection_id') and item.collection_id == target_id:
                        collection_tree.setCurrentItem(item)
                        collection_tree.scrollToItem(item)
                        return True
                    for i in range(item.childCount()):
                        if find_and_select_collection(item.child(i), target_id):
                            return True
                    return False

                # Search through all collections recursively
                for i in range(collection_tree.topLevelItemCount()):
                    item = collection_tree.topLevelItem(i)
                    if find_and_select_collection(item, current_collection.collection_id):
                        # If found, select the "selected collection" radio button
                        selected_collection_radio.setChecked(True)
                        collection_tree.setEnabled(True)
                        break

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
            previous_file = self.current_file_path if hasattr(self, 'current_file_path') else None
            previous_doc = self.pdf_document if hasattr(self, 'pdf_document') else None

            def process_pdf_item(item):
                """Process a single PDF item"""
                nonlocal analyzed_files, failed_files
                
                # Clear existing document data and session information
                self.document_data = {
                    'page_structures': {},
                    'initial_page_structures': {},
                    'metadata': {}
                }
                self.pdf_viewer.set_bounding_boxes([])
                self.pdf_viewer.pixmap = None
                self.pdf_viewer.update()
                
                # Select and scroll to the item
                self.items_tree.setCurrentItem(item)
                self.items_tree.scrollToItem(item)
                QApplication.processEvents()

                try:
                    # Open the PDF file and show first page
                    self.current_file_path = item.file_path
                    self.current_file_directory = os.path.dirname(item.file_path)
                    self.pdf_document= fitz.open(item.file_path)
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
                    if hasattr(self, 'pdf_document') and self.pdf_document:
                        try:
                            self.pdf_document.close()
                        except:
                            pass

            def process_collection(collection):
                """Process a collection and its items recursively"""
                nonlocal total_files
                
                # Select the collection to populate items tree
                self.collections_tree.setCurrentItem(collection)
                
                # Load collection items if not in cache
                if collection.collection_id not in self.collection_items_cache:
                    self.load_collection_items(collection.collection_id)
                else:
                    self._display_cached_collection_items(collection.collection_id)
                
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
                
                # Process child collections recursively
                for i in range(collection.childCount()):
                    process_collection(collection.child(i))

            # Process collections based on selection
            if selected_collection_id:
                # Find the selected collection
                def find_collection(item, target_id):
                    if hasattr(item, 'collection_id') and item.collection_id == target_id:
                        return item
                    for i in range(item.childCount()):
                        result = find_collection(item.child(i), target_id)
                        if result:
                            return result
                    return None

                # Search through all collections recursively
                selected_collection = None
                for i in range(self.collections_tree.topLevelItemCount()):
                    item = self.collections_tree.topLevelItem(i)
                    selected_collection = find_collection(item, selected_collection_id)
                    if selected_collection:
                        break

                if selected_collection:
                    process_collection(selected_collection)
                else:
                    QMessageBox.warning(self, "Collection Not Found", 
                                      "Could not find the selected collection.")
                    return
            else:
                # Process all collections
                for i in range(self.collections_tree.topLevelItemCount()):
                    process_collection(self.collections_tree.topLevelItem(i))

            # Restore previous file if any
            if previous_file and previous_doc:
                try:
                    self.current_file_path = previous_file
                    self.current_file_directory = os.path.dirname(previous_file)
                    self.pdf_document= previous_doc
                    self.current_page = 0
                    self.pdf_viewer.current_page = 0
                    self.pdf_viewer.open_pdf(previous_file)
                    self.update_navigation()
                except Exception as e:
                    logger.error(f"Error restoring previous file: {str(e)}")

            # Show completion message
            msg = f"Batch analysis completed - {analyzed_files} files analyzed, {failed_files} failed"
            self.status_label.showMessage(msg, 3000)
            QMessageBox.information(self, "Batch Analysis Complete", msg)
            logger.info(msg)

        except Exception as e:
            logger.error(f"Error in batch analysis: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error during batch analysis: {str(e)}")
            
        finally:
            # Restore cursor
            QApplication.restoreOverrideCursor()
            self.ensure_normal_cursor()

    def handle_scroll(self, value):
        """Handle scroll events from the scroll area"""
        self.pdf_viewer.update_current_page_from_scroll(value)
        self.update_navigation()

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
docker run --rm --name pdf-document-layout-analysis -p 8051:5060 --entrypoint ./start.sh huridocs/pdf-document-layout-analysis:v0.0.23

'''