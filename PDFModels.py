from peewee import *
import datetime
import os
import logging
import hashlib
import shutil

# Get user profile directory
USER_PROFILE_DIRECTORY = os.path.expanduser('~')

# Company and program info
COMPANY_NAME = "PaleoBytes"
PROGRAM_NAME = "PDFRefinery"

# Define directory structure
DEFAULT_DB_DIRECTORY = os.path.join(USER_PROFILE_DIRECTORY, COMPANY_NAME, PROGRAM_NAME)
DEFAULT_STORAGE_DIRECTORY = os.path.join(DEFAULT_DB_DIRECTORY, "data/")
DEFAULT_LOG_DIRECTORY = os.path.join(DEFAULT_DB_DIRECTORY, "logs/")
DB_BACKUP_DIRECTORY = os.path.join(DEFAULT_DB_DIRECTORY, "backups/")

# Create necessary directories
for directory in [DEFAULT_DB_DIRECTORY, DEFAULT_STORAGE_DIRECTORY, DEFAULT_LOG_DIRECTORY, DB_BACKUP_DIRECTORY]:
    os.makedirs(directory, exist_ok=True)

# Database path
DB_PATH = os.path.join(DEFAULT_DB_DIRECTORY, f"{PROGRAM_NAME.lower()}.db")

# Database setup
db = SqliteDatabase(DB_PATH)

class BaseModel(Model):
    class Meta:
        database = db

class PDFDocument(BaseModel):
    file_path = CharField(unique=True)
    file_hash = CharField()
    title = CharField(null=True)
    page_count = IntegerField()
    created_at = DateTimeField(default=datetime.datetime.now)
    last_analyzed = DateTimeField(null=True)

class PageAnalysis(BaseModel):
    document = ForeignKeyField(PDFDocument, backref='pages')
    page_number = IntegerField()
    analysis_data = TextField()  # JSON string of analysis results
    analyzed_at = DateTimeField(default=datetime.datetime.now)
    
    class Meta:
        indexes = (
            (('document', 'page_number'), True),  # Unique together
        )

def init_database(db_path):
    """Initialize the database with proper backup handling"""
    try:
        # Check if database exists
        if os.path.exists(db_path):
            # Create backup directory if it doesn't exist
            os.makedirs(DB_BACKUP_DIRECTORY, exist_ok=True)
            
            # Generate today's backup filename
            today = datetime.datetime.now().strftime('%Y%m%d%H')
            backup_filename = f"{PROGRAM_NAME.lower()}_{today}.db"
            backup_path = os.path.join(DB_BACKUP_DIRECTORY, backup_filename)
            
            # Check if today's backup exists
            if not os.path.exists(backup_path):
                try:
                    # Copy current database to backup
                    shutil.copy2(db_path, backup_path)
                    logger = logging.getLogger(PROGRAM_NAME)
                    logger.info(f"Created database backup: {backup_filename}")
                except Exception as e:
                    logger = logging.getLogger(PROGRAM_NAME)
                    logger.error(f"Failed to create database backup: {str(e)}")
        
        # Initialize database
        db.init(db_path)
        db.connect()
        db.create_tables([PDFDocument, PageAnalysis])
        logger = logging.getLogger(PROGRAM_NAME)
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger = logging.getLogger(PROGRAM_NAME)
        logger.error(f"Error initializing database: {str(e)}")
        raise
    finally:
        if not db.is_closed():
            db.close()

def calculate_file_hash(file_path):
    """Calculate SHA-256 hash of a file"""
    try:
        sha256_hash = hashlib.sha256()
        # Convert file path to absolute path and normalize
        abs_path = os.path.abspath(file_path)
        with open(abs_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger = logging.getLogger(PROGRAM_NAME)
        logger.error(f"Error calculating file hash for {file_path}: {str(e)}")
        raise 