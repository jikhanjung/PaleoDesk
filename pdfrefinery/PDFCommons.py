from PyQt6.QtWidgets import QMessageBox
import sys, os
import copy
from PyQt6.QtGui import QColor
import tempfile

COMPANY_NAME = "PaleoBytes"
PROGRAM_NAME = "PDFRefinery"
PROGRAM_VERSION = "0.0.1"
PROGRAM_AUTHOR = "PaleoBytes"
PROGRAM_COPYRIGHT = "©2025 PaleoBytes"

# Get user profile directory
USER_PROFILE_DIRECTORY = os.path.expanduser('~')

# Define directory structure
DEFAULT_DB_DIRECTORY = os.path.join( USER_PROFILE_DIRECTORY, COMPANY_NAME, PROGRAM_NAME )
DEFAULT_STORAGE_DIRECTORY = os.path.join(DEFAULT_DB_DIRECTORY, "data/")
DEFAULT_LOG_DIRECTORY = os.path.join(DEFAULT_DB_DIRECTORY, "logs/")
DB_BACKUP_DIRECTORY = os.path.join(DEFAULT_DB_DIRECTORY, "backups/")
# Database path
DATABASE_FILENAME = f"{PROGRAM_NAME.lower()}.db"
DATABASE_PATH = os.path.join(DEFAULT_DB_DIRECTORY, DATABASE_FILENAME)

# Create necessary directories
for directory in [DEFAULT_DB_DIRECTORY, DEFAULT_STORAGE_DIRECTORY, DEFAULT_LOG_DIRECTORY, DB_BACKUP_DIRECTORY]:
    os.makedirs(directory, exist_ok=True)

# get this file's directory
def get_this_file_path():
    return os.path.dirname(os.path.abspath(__file__))
    

# Get the path to the resource file
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = get_this_file_path()

    return os.path.join(base_path, relative_path)

# Define element types and their colors as class variables
ELEMENT_TYPES = ['text', 'figure', 'caption', 'title', 'authors', 'abstract', 'reference item', 'table', 'picture', 
                    'footnote', 'page header', 'page footer', 'section header', 'list item']

FIGURE_SUBTYPES = ['figure:table', 'figure:sepcimen photo', 'figure:illustration', 'figure:graph', 'figure:map', 'figure:correlation chart', 'figure:photo', 'figure:general']

# Define colors for each element type with alpha
ELEMENT_COLORS = {
    'text': QColor(0, 0, 255, 64),         # blue with alpha
    'figure': QColor(255, 0, 0, 64),       # red with alpha
    'table': QColor(0, 255, 0, 64),        # green with alpha
    'picture': QColor(255, 255, 0, 64),    # yellow with alpha
    'caption': QColor(139, 139, 0, 64),    # darkYellow with alpha
    'title': QColor(139, 0, 139, 64),   # magenta with alpha
    'authors': QColor(139, 0, 139, 64),   # magenta with alpha
    'abstract': QColor(139, 0, 139, 64),   # magenta with alpha
    'reference item': QColor(139, 0, 139, 64),   # magenta with alpha
    'footnote': QColor(139, 0, 139, 64),   # magenta with alpha
    'page header': QColor(0, 0, 139, 64),  # darkBlue with alpha
    'page footer': QColor(0, 139, 139, 64), # darkCyan with alpha
    'section header': QColor(0, 128, 0, 64), # green with alpha
    'list item': QColor(139, 0, 139, 64)   # magenta with alpha
}
