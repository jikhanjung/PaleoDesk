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
