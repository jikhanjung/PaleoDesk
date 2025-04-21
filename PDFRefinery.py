import sys
import logging
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QFileDialog, QToolBar,
                           QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                           QDialogButtonBox, QMenuBar, QMenu, QMessageBox)
from PyQt6.QtCore import Qt, QPoint, QSettings
from PyQt6.QtGui import QImage, QPixmap, QPainter, QAction, QCursor, QIcon, QPen, QColor
import fitz  # PyMuPDF
import datetime
import os
import requests
import json

COMPANY_NAME = "PaleoBytes"
PROGRAM_NAME = "PDFRefinery"
PROGRAM_VERSION = "0.0.1"
PROGRAM_AUTHOR = "Jikhan Jung"
PROGRAM_COPYRIGHT = "©2025 Jikhan Jung"

# Configure logging
def setup_logging():
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        
    # Create log filename with date
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(logs_dir, f"{PROGRAM_NAME}_{today}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(PROGRAM_NAME)
    logger.info(f"Logging initialized. Log file: {log_file}")
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
        logger.debug("Loaded settings from QSettings")
        
    def save_settings(self):
        self.settings.setValue("service/url", self.url_edit.text())
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
            matrix = fitz.Matrix(self.zoom, self.zoom)
            pix = page.get_pixmap(matrix=matrix)
            
            # Convert to QImage
            img = QImage(pix.samples, pix.width, pix.height, 
                        pix.stride, QImage.Format.Format_RGB888)
            
            # Create QPixmap
            self.pixmap = QPixmap.fromImage(img)
            self.update()
            logger.debug(f"Displayed page {self.current_page + 1} at zoom {self.zoom:.2f}")
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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{PROGRAM_NAME} v{PROGRAM_VERSION}")
        
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
        
        self.pdf_viewer = PDFViewer()
        self.setCentralWidget(self.pdf_viewer)
        
        self.create_toolbar()
        self.create_menu()
        logger.info(f"{PROGRAM_NAME} v{PROGRAM_VERSION} started")

        # Add recent files list
        self.load_recent_files()

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
        analyze_action.triggered.connect(lambda: self.analyze_pdf(True))
        toolbar.addAction(analyze_action)

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
            
            # Check for session file
            session_file = os.path.splitext(file_path)[0] + '.json'
            if os.path.exists(session_file):
                try:
                    self.load_session(session_file)
                    logger.info(f"Automatically loaded session file: {session_file}")
                    self.status_label.showMessage(f"Loaded session data for {os.path.basename(file_path)}", 3000)
                except Exception as e:
                    logger.error(f"Error loading session file {session_file}: {str(e)}")
                    QMessageBox.warning(self, "Session Load Error", 
                                      f"Could not load session file:\n{str(e)}")
                    
    def ensure_normal_cursor(self):
        """Make sure the cursor is restored to normal"""
        # Restore cursor state if it's been overridden
        while QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()

    def analyze_pdf(self, force_new_analysis=True):
        """Analyze all pages using Docker-hosted service with smart OCR decision"""
        try:
            logger.info("Analyzing all pages using Docker service")
            QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
            
            if not hasattr(self, 'doc') or not self.doc:
                logger.warning("No PDF file loaded")
                QMessageBox.warning(self, "Error", "Please open a PDF file first.")
                return
            
            # Get service URL from settings
            base_url = self.settings.value('service/url', 'http://192.168.55.253:8501').rstrip('/')
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
                # Get current input language
                input_lang = self.get_input_language()
                
                # Perform OCR
                url_ocr = f"{base_url}/ocr"
                with open(self.current_file, 'rb') as pdf_file:
                    files = {
                        'file': (os.path.basename(self.current_file), pdf_file, 'application/pdf')
                    }
                    data = {
                        'language': input_lang
                    }
                    
                    self.status_label.showMessage("Performing OCR on document...", 0)
                    QApplication.processEvents()
                    
                    try:
                        response = requests.post(url_ocr, files=files, data=data)
                        
                        if response.status_code == 200:
                            ocr_pdf_path = os.path.join(os.path.dirname(self.current_file), 
                                                      'ocr_' + os.path.basename(self.current_file))
                            
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
                        return
                    except Exception as e:
                        error_msg = f"Error during OCR: {str(e)}"
                        logger.error(error_msg)
                        QMessageBox.warning(self, "OCR Error", error_msg)
                        return
            else:
                logger.info("Sufficient text content found in PDF, proceeding with layout analysis")
                analysis_file = self.current_file
            
            # Perform layout analysis
            with open(analysis_file, 'rb') as pdf_file:
                files = {
                    'file': (os.path.basename(analysis_file), pdf_file, 'application/pdf')
                }
                
                self.status_label.showMessage("Analyzing document structure...", 0)
                logger.info("Analyzing document structure...")
                QApplication.processEvents()
                
                try:
                    response = requests.post(base_url, files=files)
                    logger.info(f"Layout analysis response: {response.status_code}")
                    
                    if response.status_code == 200:
                        results = response.json()
                        logger.info(f"Layout analysis results: {results}")
                        
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
                        
                        logger.info(f"Page elements: {page_elements}")
                        # Update the display
                        self.update_page_display()
                        
                        logger.info("Saving session after analysis")
                        # Auto-save session after analysis
                        self.save_session()
                        session_file = os.path.splitext(self.current_file)[0] + '.json'
                        if os.path.exists(session_file):
                            try:
                                self.load_session(session_file)
                            except Exception as e:
                                logger.error(f"Error loading session file {session_file}: {str(e)}")
                                QMessageBox.warning(self, "Session Load Error", 
                                                  f"Could not load session file:\n{str(e)}")
                        
                        # Show completion message
                        analyzed_pages = len(page_elements)
                        msg = f"Document analysis completed - {analyzed_pages} pages analyzed"
                        if need_ocr:
                            msg += " (with OCR)"
                            # Ask if user wants to open the OCR'd PDF
                            reply = QMessageBox.question(
                                self,
                                "OCR Complete",
                                f"OCR processing completed. The OCR'd PDF has been saved as:\n{ocr_pdf_path}\n\nWould you like to open it?",
                                QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.Yes
                            )
                            if reply == QMessageBox.Yes:
                                self.open_exported_file(ocr_pdf_path, True)
                        
                        self.status_label.showMessage(msg, 3000)
                        logger.info(msg)
                        
                    else:
                        error_msg = f"Layout analysis service returned status code: {response.status_code}"
                        logger.error(error_msg)
                        QMessageBox.warning(self, "Analysis Error", error_msg)
                        
                except requests.exceptions.ConnectionError:
                    error_msg = f"Could not connect to the layout analysis service at {base_url}. Please check the service URL in preferences."
                    logger.error(error_msg)
                    QMessageBox.warning(self, "Connection Error", error_msg)
                except Exception as e:
                    error_msg = f"Error during layout analysis: {str(e)}"
                    logger.error(error_msg)
                    logger.error("Full error details:", exc_info=True)
                    QMessageBox.warning(self, "Analysis Error", error_msg)
                    
        finally:
            # Restore cursor
            QApplication.restoreOverrideCursor()
            self.ensure_normal_cursor()

    def save_session(self):
        """Save current session data to a file"""
        try:
            if not hasattr(self, 'current_file') or not self.current_file:
                logger.debug("No current file to save session for")
                return
                
            # Get base filename without extension
            base_name = os.path.splitext(os.path.basename(self.current_file))[0]
            session_file = os.path.join(self.pdf_directory, f'{base_name}.json')
            
            # Prepare session data
            session_data = {
                'filename': self.current_file,
                'current_page': self.current_page,
                'document_data': {
                    'page_structures': self.document_data['page_structures'],
                    'metadata': self.document_data['metadata'],
                    'page_dimensions': self.document_data.get('page_dimensions', {})
                },
                'timestamp': datetime.datetime.now().isoformat(),
                'total_pages': len(self.doc) if self.doc else 0,
                'session_info': {
                    'analyzed_pages': len(set(self.document_data['page_structures'].keys())),
                    'app_version': '0.0.1'
                }
            }
            
            # Save session data as JSON
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Session saved to {session_file}")
            logger.debug(f"Saved {len(self.document_data['page_structures'])} page structures")
            
        except Exception as e:
            logger.error(f"Error saving session: {str(e)}")
            logger.error("Full error details:", exc_info=True)

    def update_page_display(self):
        """Update the PDF display and text areas with the current page content"""
        logger.debug(f"Updating page display: current page={self.current_page}")
        try:
            if self.doc is None:
                return
            
            # Update PDF display
            page = self.doc.load_page(self.current_page)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(4, 4))  # 2x zoom for better quality
            image = QImage(pixmap.samples, pixmap.width, pixmap.height, pixmap.stride, QImage.Format.Format_RGB888)
            self.pdf_viewer.set_page(self.current_page, QPixmap.fromImage(image))

            # Get the page structure if it exists
            if str(self.current_page) in self.document_data['page_structures']:
                structure = self.document_data['page_structures'][str(self.current_page)]
                self.pdf_viewer.set_page_structure(self.current_page, structure)
                logger.debug(f"Setting page structure for page {self.current_page}: {structure}")
            else:
                self.pdf_viewer.set_page_structure(self.current_page, None)
                logger.debug(f"No structure found for page {self.current_page}, {self.document_data['page_structures']}")
            
            # Update page number display
            self.current_page_input.setText(str(self.current_page + 1))
            self.total_pages_label.setText(f"/ {len(self.doc)}")
            
            # Extract and display text
            self.extract_text()
            
            
            # Update structure text if available
            structure = self.get_page_structure(self.current_page)
            if structure:
                # Display structure in the structure text area
                self.show_page_structure(structure)
            
            # Update button states
            self.update_buttons()
            
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
        if self.doc:
            self.current_page_input.setText(str(self.current_page + 1))
            self.total_pages_label.setText(f"/ {len(self.doc)}")
            self.prev_button.setEnabled(self.current_page > 0)
            self.next_button.setEnabled(self.current_page < len(self.doc) - 1)

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

    def load_session(self, session_file=None):
        """Load a previously saved session"""
        try:
            if not session_file:
                session_file, _ = QFileDialog.getOpenFileName(
                    self, "Load Session", "", "Session Files (*.json)"
                )
                if not session_file:
                    return

            logger.info(f"Loading session from {session_file}")
            
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
                
            # Load document data
            self.document_data = session_data.get('document_data', {})
            logger.debug(f"Loaded document data: {self.document_data.keys()}")
            
            # Load page structures
            page_structures = dict(self.document_data.get('page_structures', {}))
            logger.debug(f"Loaded page structures: {page_structures}")
            
            # Load PDF file with fallback options
            pdf_path = session_data.get('filename')
            if not pdf_path:
                logger.error("No PDF filename found in session file")
                QMessageBox.warning(self, "Error", "No PDF filename found in session file")
                return
                
            # Try to find the PDF file
            if not os.path.exists(pdf_path):
                logger.warning(f"PDF file not found at original path: {pdf_path}")
                
                # Try to find PDF in the same directory as the session file
                session_dir = os.path.dirname(session_file)
                pdf_filename = os.path.basename(pdf_path)
                local_pdf_path = os.path.join(session_dir, pdf_filename)
                
                if os.path.exists(local_pdf_path):
                    pdf_path = local_pdf_path
                    logger.info(f"Found PDF file in session directory: {pdf_path}")
                else:
                    # Ask user to locate the PDF file
                    pdf_path, _ = QFileDialog.getOpenFileName(
                        self,
                        "Locate PDF File", 
                        session_dir, 
                        "PDF Files (*.pdf)"
                    )
                    if not pdf_path:
                        logger.error("User cancelled PDF file selection")
                        QMessageBox.warning(self, "Error", "PDF file is required to load the session")
                        return
                    logger.info(f"User selected PDF file: {pdf_path}")
            
            # Open the PDF file
            try:
                self.current_file = pdf_path
                self.pdf_directory = os.path.dirname(pdf_path)
                self.doc = fitz.open(pdf_path)
                self.current_page = session_data.get('current_page', 0)
                self.pdf_viewer.current_page = self.current_page
                self.pdf_viewer.open_pdf(pdf_path)
                
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
                
                logger.info(f"Successfully loaded session with {len(self.doc)} pages")
                self.status_label.showMessage("Session loaded successfully", 3000)
                
            except Exception as e:
                logger.error(f"Error opening PDF file: {str(e)}")
                QMessageBox.warning(self, "Error", f"Failed to open PDF file: {str(e)}")
                return
                
        except Exception as e:
            logger.error(f"Error loading session: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            QMessageBox.warning(self, "Error", f"Failed to load session: {str(e)}")
            
    def update_navigation(self):
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