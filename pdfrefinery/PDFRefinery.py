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
                           QFrame, QStatusBar, QProgressBar, QCheckBox, QGroupBox, QToolTip,
                           QStyle, QColorDialog)
from PyQt6.QtCore import Qt, QPoint, QPointF, QSettings, QSize, QRect, QRectF, pyqtSignal, QTimer, QEvent, QCoreApplication
from PyQt6.QtGui import QImage, QPixmap, QPainter, QAction, QCursor, QIcon, QPen, QColor, QBrush, QWheelEvent, QPainterPath
import fitz  # PyMuPDF
import datetime
import os
import requests
import json
import sqlite3
import shutil
from peewee import DoesNotExist, chunked
from peewee_migrate import Router
from PDFCommons import *
from PrComponents import *
from PrDialogs import *
from PDFModels import (db, PDFDocument, PageAnalysis, SessionData, StructuredElement, calculate_file_hash, PrFigure)                       
import hashlib
import uuid
import copy
import io
from PIL import Image

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

def get_analysis_done_icon():
    # Get the application style (requires QApplication to be initialized)
    style = QApplication.instance().style()
    pixmap = style.standardPixmap(QStyle.StandardPixmap.SP_DialogApplyButton)
    return QIcon(pixmap)



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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{PROGRAM_NAME} v{PROGRAM_VERSION}")
        
        # Initialize database
        self.prepare_database()
        
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
        
        # Create right panel splitter
        self.right_splitter = QSplitter(Qt.Orientation.Horizontal)  # Made it a member variable
        main_splitter.addWidget(self.right_splitter)
        
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
        
        self.pdf_viewer = PDFViewer(self)
        pdf_layout.addWidget(self.pdf_viewer)
        
        self.pdf_scroll.setWidget(pdf_container)
        self.right_splitter.addWidget(self.pdf_scroll)
        
        # Create structured content view with scroll area
        self.content_scroll = QScrollArea()
        self.content_scroll.setWidgetResizable(True)
        self.structured_view = StructuredContentView(self)
        self.content_scroll.setWidget(self.structured_view)
        self.right_splitter.addWidget(self.content_scroll)
        
        # Create dummy widget
        self.dummy_widget = QWidget()
        dummy_layout = QVBoxLayout(self.dummy_widget)
        dummy_label = QLabel("Dummy Content")
        dummy_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dummy_layout.addWidget(dummy_label)
        self.right_splitter.addWidget(self.dummy_widget)
        self.dummy_widget.hide()
        
        # Create toggle button for dummy widget
        self.dummy_toggle_btn = QPushButton("◀", self)
        self.dummy_toggle_btn.setFixedSize(20, 200)  # Tall and narrow button
        self.dummy_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 2px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.dummy_toggle_btn.clicked.connect(self.toggle_dummy_widget)
        self.dummy_toggle_btn.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)  # Keep button on top
        self.dummy_toggle_btn.raise_()  # Ensure button is on top
        self.dummy_toggle_btn.show()
        logger.debug("Created dummy toggle button")
        
        # Set initial splitter sizes (40% for library, 30% for PDF, 30% for content)
        main_splitter.setSizes([400, 800])
        # Set right splitter sizes to give more space to PDF viewer and content view
        self.right_splitter.setSizes([400, 400, 400])  # Adjusted sizes for better proportions
        
        self.create_toolbar()
        self.create_menu()
        logger.info(f"{PROGRAM_NAME} v{PROGRAM_VERSION} started")

        # Add recent files list
        self.load_recent_files()

        # Initialize collection items cache
        self.collection_items_cache = {}

        # Connect PDFViewer page change to StructuredContentView scroll
        self.pdf_viewer.currentPageChanged.connect(self.structured_view.scroll_to_page_element)

        # Show the toggle buttons
        self.library_toggle_btn.show()
        self.update_library_toggle_button_position()
        self.update_dummy_toggle_button_position()
        logger.debug("Initialized toggle buttons")

    def prepare_database(self):
        try:

            migrations_path = resource_path("migrations")
            logger.info("migrations path: %s", migrations_path)
            logger.info("database path: %s", DATABASE_PATH)
            now = datetime.datetime.now()
            date_str = now.strftime("%Y%m%d")

            # backup database file to backup directory
            backup_path = os.path.join( DB_BACKUP_DIRECTORY, DATABASE_FILENAME + '.' + date_str )
            if not os.path.exists(backup_path) and os.path.exists(DATABASE_PATH):
                shutil.copy2(DATABASE_PATH, backup_path)
                logger.info("backup database to %s", backup_path)
                # read backup directory and delete old backups
                backup_list = os.listdir(DB_BACKUP_DIRECTORY)
                # filter out non-backup files
                backup_list = [f for f in backup_list if f.startswith(DATABASE_FILENAME)]
                backup_list.sort()
                if len(backup_list) > 10:
                    for i in range(len(backup_list) - 10):
                        os.remove(os.path.join(DB_BACKUP_DIRECTORY, backup_list[i]))                    
            
            #logger.info("database name: %s", mu.DEFAULT_DATABASE_NAME)
            #print("migrations path:", migrations_path)
            db.connect()
            router = Router(db, migrate_dir=migrations_path)

            # Auto-discover and run migrations
            router.run()        
            return
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
        self.collections_tree.setStyleSheet("""
            QTreeWidget::item:selected {
                background-color: #2196F3;
                color: white;
            }
            QTreeWidget::item:selected:hover {
                background-color: #2196F3;
                color: white;
            }
            QTreeWidget::item:hover {
                background-color: #E3F2FD;
            }
        """)
        
        # Create items tree widget
        self.items_tree = QTreeWidget()
        self.items_tree.setHeaderLabels(["Items"])
        self.items_tree.setColumnWidth(0, 300)
        self.items_tree.setHeaderHidden(True)  # Hide header
        self.items_tree.itemClicked.connect(self.item_clicked)
        self.items_tree.setStyleSheet("""
            QTreeWidget::item:selected {
                background-color: #2196F3;
                color: white;
            }
            QTreeWidget::item:selected:hover {
                background-color: #2196F3;
                color: white;
            }
            QTreeWidget::item:hover {
                background-color: #E3F2FD;
            }
        """)
        
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
        
        # Create toggle button for library dock
        self.library_toggle_btn = QPushButton("◀", self)
        self.library_toggle_btn.setFixedSize(20, 200)  # Tall and narrow button
        self.library_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 2px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.library_toggle_btn.clicked.connect(self.toggle_library_dock)
        self.library_toggle_btn.raise_()  # Ensure button is on top
        self.library_toggle_btn.show()
        
        # Position the buttons
        self.update_library_toggle_button_position()
        
        logger.info("Directory tree widgets created with splitter")

    def toggle_library_dock(self):
        """Toggle the visibility of the library dock"""
        if self.library_dock.isVisible():
            self.library_dock.hide()
            self.library_toggle_btn.setText("▶")
        else:
            self.library_dock.show()
            self.library_toggle_btn.setText("◀")
        self.update_library_toggle_button_position()

    def update_library_toggle_button_position(self):
        """Update the position of the library toggle button"""
        if not hasattr(self, 'library_toggle_btn') or not self.library_toggle_btn:
            return
            
        # Always position at the left edge of the window
        x = 0
        y = (self.height() - self.library_toggle_btn.height()) // 2
            
        self.library_toggle_btn.move(x, y)
        self.library_toggle_btn.raise_()  # Ensure button is on top
        self.library_toggle_btn.show()

    def update_dummy_toggle_button_position(self):
        """Update the position of the dummy toggle button"""
        logger.debug("Updating dummy toggle button position")
        if not hasattr(self, 'dummy_toggle_btn') or not self.dummy_toggle_btn:
            logger.debug("Dummy toggle button not initialized")
            return
            
        # Always position at the right edge of the window
        x = self.width() - self.dummy_toggle_btn.width()
        y = (self.height() - self.dummy_toggle_btn.height()) // 2
        logger.debug(f"Window size: {self.width()}x{self.height()}")
            
        self.dummy_toggle_btn.move(x, y)
        self.dummy_toggle_btn.raise_()  # Ensure button is on top
        self.dummy_toggle_btn.show()
        logger.debug(f"Moved dummy toggle button to MainWindow position ({x}, {y})")

    def toggle_dummy_widget(self):
        """Toggle the visibility of the dummy widget"""
        if self.dummy_widget.isVisible():
            self.dummy_widget.hide()
            self.dummy_toggle_btn.setText("◀")
        else:
            self.dummy_widget.show()
            self.dummy_toggle_btn.setText("▶")
        self.update_dummy_toggle_button_position()

    def resizeEvent(self, event):
        """Handle window resize events"""
        logger.debug("[MainWindow::resizeEvent] Window resize event triggered")
        super().resizeEvent(event)
        self.update_library_toggle_button_position()
        self.update_dummy_toggle_button_position()

    def collection_clicked(self, item, column):
        """Handle click on collection item"""
        if hasattr(item, 'collection_id'):
            # Clear the items tree
            self.items_tree.clear()
            
            # Clear the PDF viewer
            self.pdf_viewer.clear_document()
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
            logger.debug("[MainWindow::item_clicked] Item clicked: %s", item.text(0))
            # Set wait cursor
            QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
            
            zotero_key = None
            if hasattr(item, 'file_path') and item.file_path:
                # Direct click on a PDF item
                if item.file_path.lower().endswith('.pdf'):
                    if hasattr(item, 'zotero_key'):
                        zotero_key = item.zotero_key
                    logger.debug("[MainWindow::item_clicked] Loading PDF file: %s", item.file_path)
                    self.load_pdf_file(item.file_path, zotero_key)
            else:
                # Click on a parent item - check for PDF attachments
                pdf_items = []
                for i in range(item.childCount()):
                    child = item.child(i)
                    if (hasattr(child, 'file_path') and 
                        child.file_path and 
                        child.file_path.lower().endswith('.pdf')):
                        pdf_items.append(child)
                
                pdf_item = pdf_items[0]
                if hasattr(pdf_item, 'zotero_key'):
                    zotero_key = pdf_item.zotero_key
                self.load_pdf_file(pdf_item.file_path, zotero_key)
                logger.debug(f"Loading single PDF attachment: {pdf_item.file_path}")

        finally:
            # Restore cursor
            QApplication.restoreOverrideCursor()
            self.ensure_normal_cursor()

    def load_pdf_file(self, file_path, zotero_key=None):
        """Load a PDF file and try to load its analysis from database"""
        self.current_file_path = file_path
        self.current_file_directory = os.path.dirname(file_path)
        self.document_data = {
            'page_structures': {},
            'initial_page_structures': {},
            'metadata': {}
        }
        self.pdf_viewer.clear_document()

        # If no Zotero key found in tree, try to extract from path
        if not zotero_key and 'storage' in file_path:
            storage_dir = os.path.dirname(file_path)
            potential_key = os.path.basename(storage_dir)
            if len(potential_key) == 8:  # Zotero keys are 8 characters
                zotero_key = potential_key
                logger.debug(f"Extracted Zotero key from path: {zotero_key}")
        self.current_zotero_key = zotero_key
        
        self.pdf_document = fitz.open(file_path)
        self.current_page = 0
        self.pdf_viewer.set_document(self.pdf_document)
        self.structured_view.set_document(self.pdf_document)
        self.update_navigation()

        self.status_label.showMessage(f"Opened: {os.path.basename(file_path)}", 3000)
        logger.info(f"Opened PDF file: {self.current_file_path}")

        try:
            self.load_document_record()
            logger.info(f"Loaded document data for {self.current_file_path}")
            self.load_session_record()
            logger.info(f"Loaded session data for {self.current_file_path}")
        except Exception as e:
            logger.error(f"Error loading document record: {str(e)}")

        try:
            self.update_page_display()
            # Update structured content view with all page structures
            #self.structured_view.update_content(self.document_data['page_structures'])
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
        
        # Reset Page button
        reset_page_action = QAction("Reset Page", self)
        reset_page_action.triggered.connect(self.pdf_viewer.reset_page)
        toolbar.addAction(reset_page_action)
        
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

        # extract figures action
        extract_figures_action = QAction("Extract Figures", self)
        extract_figures_action.triggered.connect(self.extract_figures)
        toolbar.addAction(extract_figures_action)

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
            self.pdf_viewer.set_document(self.pdf_document)
            self.update_navigation()
            self.status_label.showMessage(f"Opened: {os.path.basename(file_path)}", 3000)
            logger.info(f"Opened PDF file: {self.current_file_path}")
            
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

    def analyze_pdf(self, file_path, force_analysis=False):
        """Analyze the PDF file and store results in the database."""
        try:
            document = self.document_record
            logger.info(f"document {document}")

            session = document.sessions.order_by(SessionData.last_accessed.desc()).first()
            
            if session and not force_analysis:
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

                logger.info(f"need_ocr {need_ocr} document {document}")

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

                logger.info(f"Going to do layout analysis for document {document}")
                
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
                                page_num = int(element['page_number'] - 1)  # Convert to 0-based index
                                if str(page_num) not in page_elements:
                                    page_elements[str(page_num)] = []
                                
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
                                    'id': len(page_elements[str(page_num)]),  # ID starts from 0 for each page
                                    'page_number': int(page_num)
                                }
                                page_elements[str(page_num)].append(structured_element)
                            
                            # Store the analysis results
                            for page_num, elements in page_elements.items():
                                self.document_data['page_structures'][str(page_num)] = {
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
                                self.document_data['initial_page_structures'][str(page_num)] = {
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
                            
                            # Update document in database with analysis timestamp
                            logger.info(f"Updating document {document} in database with analysis timestamp")
                            with db:
                                document.last_analyzed = datetime.datetime.now()
                                document.save()
                            
                            # Update icon in tree
                            current_item = self.items_tree.currentItem()
                            icon = get_analysis_done_icon()
                            current_item.setIcon(0, icon)
                            
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

    def save_session(self):
        """Save current session data to database"""
        try:
            if not hasattr(self, 'current_file_path') or not self.current_file_path:
                logger.debug("No current file to save session for")
                return

            document = self.document_record

            with db:
                # Update document fields if needed
                if document:
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
                                    'element_id': int(element.get('id', '')),
                                    'element_type': element.get('category', 'unknown'),
                                    'coordinates': json.dumps(element.get('coordinates', [])),
                                    'content': json.dumps(element.get('content', {})),
                                    'caption': json.dumps(element.get('caption', {})),
                                    'metadata': json.dumps(element.get('metadata', {})),
                                    'created_at': datetime.datetime.now(),
                                    'updated_at': datetime.datetime.now()
                                })
                    # printout elements' page and id, not the whole element for elements_to_create
                    #logger.info(f"Elements to create: {[{'page': e['page_number'], 'id': e['element_id']} for e in elements_to_create]}")

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
                        'zotero_key': self.current_zotero_key
                    }
                }
                
                # Create new session entry
                SessionData.create(
                    document=document,
                    current_page=self.current_page,
                    session_data=json.dumps(session_data, ensure_ascii=False),
                    last_accessed=datetime.datetime.now()
                )
                
                logger.info(f"Session saved to database for {self.current_file_path} (Zotero key: {self.current_zotero_key})")
                logger.debug(f"Saved {len(self.document_data['page_structures'])} page structures")
                
        except Exception as e:
            logger.error(f"Error saving session: {str(e)}")
            logger.error("Full error details:", exc_info=True)

    def load_document_record(self):
        """Load document record from database"""
        self.document_record = None
        try:
            if self.current_zotero_key:
                self.document_record = PDFDocument.get(PDFDocument.zotero_key == self.current_zotero_key)

            if not self.document_record:
                self.document_record = PDFDocument.get(PDFDocument.file_path == self.current_file_path)

            logger.info(f"Loaded document record for {self.current_file_path}")
        except Exception as e:
            logger.error(f"Error loading document record: {str(e)}")

    def load_session_record(self):
        """Load a previously saved session from database"""
        try:
            logger.info(f"Loading session for {self.current_file_path}")

            self.session_record = None
            if self.document_record:
                self.session_record = self.document_record.sessions.order_by(SessionData.created_at.desc()).first()

            if not self.session_record:
                logger.error(f"No session record found for {self.current_file_path}")
                return

            # Load session data
            session_data = json.loads(self.session_record.session_data)
            self.document_data = session_data.get('document_data', {})
            logger.debug(f"Loaded document data: {self.document_data.keys()}")
            
            # Load page structures
            page_structures = dict(self.document_data.get('page_structures', {}))
            #logger.debug(f"Loaded page structures: {page_structures}")

            self.element_records = []

            self.element_records = self.document_record.elements.order_by(StructuredElement.page_number, StructuredElement.element_id)
            
            logger.debug(f"Loading {len(self.element_records)} structured elements")
            for element in self.element_records:
                page_num = int(element.page_number)
                page_key = str(page_num)
                if page_key not in page_structures:
                    page_structures[page_key] = {'structure': {'elements': []}}

                element_data = element.to_dict()
                
                # Update existing element or append new one
                elements_list = page_structures[page_key]['structure']['elements']
                for i, existing in enumerate(elements_list):
                    if int(existing.get('id')) == int(element.element_id):
                        elements_list[i] = element_data
                        break
                else:
                    elements_list.append(element_data)
            
            self.update_navigation()
                
            logger.info(f"Successfully loaded session with {len(self.pdf_document)} pages (Zotero key: {self.current_zotero_key})")
            self.status_label.showMessage("Session loaded successfully", 3000)
            
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
            #self.pdf_viewer.display_all_pages()
            logger.debug(f"Displaying page {self.current_page}, {self.document_data}")

            # Set bounding boxes for all pages
            if 'page_structures' in self.document_data:
                logger.debug(f"Page structures: {self.document_data['page_structures']}")
                all_boxes = {}
                for page_num, structure in self.document_data['page_structures'].items():
                    boxes = structure.get('structure', {}).get('elements', [])
                    all_boxes[int(page_num)] = boxes
                self.pdf_viewer.set_bounding_boxes(all_boxes)
                logger.debug(f"Setting bounding boxes for all pages: {len(all_boxes)} pages")
                
                # Update structured content view
                #self.structured_view.update_content(self.document_data['page_structures'])
            else:
                self.pdf_viewer.set_bounding_boxes({})
                logger.debug("No page structures found")
            
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
        """Go to previous page"""
        if self.current_page > 0:
            # Get current page height
            page_height = self.pdf_viewer.page_pixmaps[self.current_page]['height']
            # Calculate scroll position for previous page
            scroll_pos = self.pdf_scroll.verticalScrollBar().value() - page_height
            # Update current page
            self.current_page -= 1
            self.pdf_viewer.current_page = self.current_page
            # Scroll to new position
            self.pdf_scroll.verticalScrollBar().setValue(scroll_pos)
            # Update navigation
            self.update_navigation()
            logger.debug(f"Navigated to previous page: {self.current_page + 1}")
            
    def next_page(self):
        if self.pdf_document and self.current_page < len(self.pdf_document) - 1:
            self.current_page += 1
            self.pdf_viewer.current_page = self.current_page
            #self.pdf_viewer.display_all_pages()
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
                #self.pdf_viewer.display_all_pages()
                
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
            logger.debug(f"Trying to find parent items for collection {collection_id}")
            
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
                icon = None
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
                        
                        # If there's a parent item, update its icon based on all child PDFs
                        if parent_item is not None:
                            # Don't set parent icon here - it will be handled by the cache display
                            pass
 
                        # Save document to database only if it doesn't exist
                        try:
                            with db:
                                try:
                                    document = PDFDocument.get(PDFDocument.zotero_key == zotero_key)
                                    if len(document.sessions) > 0:
                                        logger.info(f"Sessions: {document.sessions}")
                                        icon = get_analysis_done_icon()
                                        pdf_item.setIcon(0, icon)
                                        logger.info(f"Session already exists in database: {zotero_key}")

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
                        
                        return True, icon
                    else:
                        logger.warning(f"No PDF files found in directory: {storage_dir}")
                else:
                    logger.warning(f"Storage directory not found: {storage_dir}")
                return False, icon
            
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
                        'children': [],
                        'icon': None
                    }
                    
                    # Track if any child has been analyzed
                    any_child_analyzed = False
                    
                    # Add PDF attachments as children
                    for attachment in attachments_by_parent[item_id]:
                        try:
                            zotero_key = attachment[4]  # Zotero key
                            parent_key = attachment[6]  # Parent's Zotero key
                            display_name = attachment[3]
                            if display_name.startswith('storage:'):
                                display_name = os.path.basename(display_name)
                            
                            add_okay, icon = add_pdf_to_items_tree(zotero_key, display_name, parent_item, collection_id, parent_key)
                            if add_okay:
                                # Store child item in cache
                                child_item_data = {
                                    'text': display_name,
                                    'zotero_key': zotero_key,
                                    'parent_zotero_key': parent_key,
                                    'file_path': os.path.join(storage_base, zotero_key, [f for f in os.listdir(os.path.join(storage_base, zotero_key)) if f.lower().endswith('.pdf')][0]),
                                    'icon': icon
                                }
                                parent_item_data['children'].append(child_item_data)
                                
                                # Check if this child is analyzed
                                if icon is not None:
                                    any_child_analyzed = True
                                    logger.info(f"Child item data: {display_name} {zotero_key} {parent_key} {icon}")
                                
                        except Exception as e:
                            logger.error(f"Error processing attachment {zotero_key}: {str(e)}")
                    
                    # Set parent icon if any child is analyzed
                    if any_child_analyzed:
                        parent_item.setIcon(0, self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
                        parent_item_data['icon'] = self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
                    #logger.info(f"Parent item data: {parent_item_data}")
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
                    
                    added_okay, icon = add_pdf_to_items_tree(zotero_key, display_name, None, attachment[5])
                    if added_okay:  # attachment[5] is collectionID
                        # Store standalone item in cache
                        standalone_item_data = {
                            'text': display_name,
                            'zotero_key': zotero_key,
                            'file_path': os.path.join(storage_base, zotero_key, [f for f in os.listdir(os.path.join(storage_base, zotero_key)) if f.lower().endswith('.pdf')][0]), 
                            'icon': icon
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
            QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
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
                any_child_analyzed = False
                for child_data in item_data.get('children', []):
                    child_item = QTreeWidgetItem(parent_item)
                    child_item.setText(0, child_data['text'])
                    child_item.zotero_key = child_data['zotero_key']
                    if 'parent_zotero_key' in child_data:
                        child_item.parent_zotero_key = child_data['parent_zotero_key']
                    if 'file_path' in child_data:
                        child_item.file_path = child_data['file_path']
                    # Set child icon based on analysis status
                    icon = child_data['icon']#self.get_analysis_status_icon(child_data['zotero_key'])
                    if icon is not None:  # Only set icon if not None
                        child_item.setIcon(0, icon)
                        any_child_analyzed = True
                
                # Set parent icon if any child is analyzed
                #if any_child_analyzed and 'zotero_key' in item_data:
                if item_data['icon'] is not None:
                    parent_item.setIcon(0, item_data['icon'])
            
            self.status_label.showMessage("Loaded collection items from cache", 3000)
            logger.debug(f"Displayed cached items for collection {collection_id}")
            # restore cursor
            QApplication.restoreOverrideCursor()
            
        except Exception as e:
            logger.error(f"Error displaying cached collection items: {str(e)}")
            # restore cursor
            QApplication.restoreOverrideCursor()
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
                    self.pdf_viewer.set_document(self.pdf_document)
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
                    self.pdf_document.close()

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
                    self.pdf_viewer.set_document(self.pdf_document)
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

    def extract_figures(self):
        """Extract figures from the current document and store them in the PrFigure table as PNG binary."""
        if not self.pdf_document or not self.document_record:
            QMessageBox.warning(self, "No Document", "No PDF document is loaded.")
            return
        try:

            def get_pixmap(page_number, coordinates):
                zoom = 3.0
                matrix = fitz.Matrix(zoom, zoom)
                page = self.pdf_document[int(page_number)]
                pix = page.get_pixmap(matrix=matrix)
                # Convert relative coordinates to absolute pixel values
                x1 = int(coordinates[0]['x'] * pix.width)
                y1 = int(coordinates[0]['y'] * pix.height)
                x2 = int(coordinates[2]['x'] * pix.width)
                y2 = int(coordinates[2]['y'] * pix.height)
                # Ensure coordinates are within bounds
                x1, x2 = max(0, min(x1, x2)), min(pix.width, max(x1, x2))
                y1, y2 = max(0, min(y1, y2)), min(pix.height, max(y1, y2))
                if x2 <= x1 or y2 <= y1:
                    return None
                # Crop the image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                cropped = img.crop((x1, y1, x2, y2))
                # Save as PNG to bytes
                buf = io.BytesIO()
                cropped.save(buf, format="PNG")
                return buf.getvalue()

            # Remove previous figures for this document
            PrFigure.delete().where(PrFigure.document == self.document_record).execute()
            page_structures = self.document_data.get('page_structures', {})
            figure_count = 0
            for page_num, structure in page_structures.items():
                elements = structure.get('structure', {}).get('elements', [])
                for element in elements:
                    category = element.get('category', '').lower()
                    if not category.startswith('figure:'):
                        continue
                    coords = element.get('coordinates', [])
                    if len(coords) < 4:
                        continue
                    # Find caption (if linked)
                    caption_text = None
                    caption_pixmap = None
                    linked_elements = element.get('linked_elements', None)
                    if linked_elements:
                        for ref in linked_elements:
                            ref_page, ref_id = ref
                            # Find the linked element
                            for e in page_structures.get(str(ref_page), {}).get('structure', {}).get('elements', []):
                                if int(e.get('id')) == int(ref_id) and e.get('category', '').lower() == 'caption':
                                    caption_text = e.get('content', {}).get('text', None)
                                    caption_page_num = ref_page
                                    caption_element = e
                                    caption_coords = e.get('coordinates', [])
                                    break
                            if caption_text:
                                break

                    figure_binary = get_pixmap(page_num, coords)
                    caption_binary = get_pixmap(caption_page_num, caption_coords)

                    # Insert into PrFigure
                    PrFigure.create(
                        document=self.document_record,
                        figure_number=str(figure_count + 1),
                        figure_page_number=int(page_num) + 1,
                        figure_binary=figure_binary,
                        caption_text=caption_text,
                        caption_binary=caption_binary
                    )
                    figure_count += 1
            if hasattr(self, 'structured_view'):
                self.structured_view.show_figures_from_db(self.document_record)

            QMessageBox.information(self, "Figures Extracted", f"Extracted {figure_count} figures to the database.")
        except Exception as e:
            logger.error(f"Error extracting figures: {str(e)}")
            QMessageBox.critical(self, "Extraction Error", f"Error extracting figures: {str(e)}")

    def handle_scroll(self, value):
        """Handle scroll events from the PDF viewer"""
        logger.debug(f"[MainWindow] handle_scroll: value={value}")
        # Update current page in PDF viewer
        self.pdf_viewer.update_current_page(value)
        logger.debug(f"[MainWindow] handle_scroll: current_page={self.pdf_viewer.current_page} {self.current_page}")
        
        # Update current page in main window to match PDF viewer
        if self.pdf_viewer.current_page != self.current_page:
            self.current_page = self.pdf_viewer.current_page
            # Update page display in toolbar
            self.current_page_input.setText(str(self.current_page + 1))
            self.update_navigation()
            logger.debug(f"Page updated to {self.current_page + 1} from scroll")

    def get_analysis_status_icon(self, zotero_key):
        """Get the appropriate icon for the document's analysis status"""
        try:
            with db:
                try:
                    document = PDFDocument.get(PDFDocument.zotero_key == zotero_key)
                    
                    # Check if document has been analyzed
                    if document.last_analyzed:
                        return self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
                    
                    # Check if document has session data
                    session = (SessionData
                             .select()
                             .where(SessionData.document == document)
                             .order_by(SessionData.last_accessed.desc())
                             .first())
                    
                    if session:
                        return self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
                    
                    return None  # No icon for unanalyzed documents
                except DoesNotExist:
                    return None  # No icon for documents not in database
        except Exception as e:
            logger.error(f"Error getting analysis status icon: {str(e)}")
            return None  # No icon on error


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