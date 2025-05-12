from peewee import *
import datetime
import os
import logging
import hashlib
import shutil
from PDFCommons import *
import json

# Initialize logger
logger = logging.getLogger(PROGRAM_NAME)


# Database setup
db = SqliteDatabase(DATABASE_PATH)

class BaseModel(Model):
    class Meta:
        database = db

class PDFDocument(BaseModel):
    """Model for storing PDF document information"""
    file_path = CharField(unique=True)
    file_hash = CharField(null=True)  # Make file_hash nullable since we might use zotero_key instead
    zotero_key = CharField(null=True)  # Add Zotero key field
    title = CharField()
    page_count = IntegerField()
    last_analyzed = DateTimeField(null=True)
    created_at = DateTimeField(default=datetime.datetime.now)
    updated_at = DateTimeField(default=datetime.datetime.now)

    class Meta:
        indexes = (
            # Create non-unique indexes
            (('file_hash',), False),
            (('zotero_key',), False),
        )

class PageAnalysis(BaseModel):
    document = ForeignKeyField(PDFDocument, backref='pages')
    page_number = IntegerField()
    analysis_data = TextField()  # JSON string of analysis results
    analyzed_at = DateTimeField(default=datetime.datetime.now)
    
    class Meta:
        indexes = (
            (('document', 'page_number'), True),  # Unique together
        )

class SessionData(BaseModel):
    document = ForeignKeyField(PDFDocument, backref='sessions')
    current_page = IntegerField()
    session_data = TextField()  # JSON string of session data
    created_at = DateTimeField(default=datetime.datetime.now)
    last_accessed = DateTimeField(default=datetime.datetime.now)
    
    class Meta:
        indexes = (
            (('document', 'created_at'), False),  # Index for faster lookups
        )

class StructuredElement(BaseModel):
    """Model for storing structured elements from PDFs"""
    document = ForeignKeyField(PDFDocument, backref='elements')
    page_number = IntegerField()
    element_id = CharField()  # ID within the page
    element_type = CharField()  # Type of element (e.g., 'figure', 'table', 'text')
    coordinates = TextField()  # JSON string of coordinates
    content = TextField(null=True)  # Text content or description
    caption = TextField(null=True)  # Associated caption text
    metadata = TextField(null=True)  # JSON string for additional metadata
    created_at = DateTimeField(default=datetime.datetime.now)
    updated_at = DateTimeField(default=datetime.datetime.now)
    
    class Meta:
        indexes = (
            (('document', 'page_number', 'element_id'), True),  # Unique together
            (('document', 'element_type'), False),  # Index for type-based queries
        )

    def to_dict(self):
        # Convert element to dictionary format
        coords = json.loads(self.coordinates)
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
            'id': self.element_id,
            'category': self.element_type,
            'coordinates': normalized_coords,
            'content': json.loads(self.content) if self.content else {},
            'caption': json.loads(self.caption) if self.caption else {},
            'metadata': json.loads(self.metadata) if self.metadata else {}
        }
        return element_data

def init_database(db_path):
    """Initialize the database with all required tables"""
    try:
        db.init(db_path)
        with db:
            # Check if tables exist
            tables_exist = db.get_tables()
            
            if not tables_exist:
                # Create tables if they don't exist
                db.create_tables([PDFDocument, PageAnalysis, SessionData, StructuredElement])
                logger.info("Created new database tables")
            else:
                # Handle migration for existing database
                try:
                    # Check existing columns
                    cursor = db.execute_sql(
                        "SELECT name FROM pragma_table_info('pdfdocument')"
                    )
                    existing_columns = [row[0] for row in cursor.fetchall()]
                    
                    # Create new table with all required columns
                    db.execute_sql("""
                        CREATE TABLE IF NOT EXISTS pdfdocument_new (
                            id INTEGER PRIMARY KEY,
                            file_path VARCHAR(255) NOT NULL UNIQUE,
                            file_hash VARCHAR(255),
                            zotero_key VARCHAR(255),
                            title VARCHAR(255) NOT NULL,
                            page_count INTEGER NOT NULL,
                            last_analyzed DATETIME,
                            created_at DATETIME NOT NULL,
                            updated_at DATETIME NOT NULL
                        )
                    """)
                    
                    # Copy data from old table to new table
                    if 'updated_at' in existing_columns:
                        db.execute_sql("""
                            INSERT INTO pdfdocument_new 
                            SELECT id, file_path, file_hash, zotero_key, title, page_count,
                                   last_analyzed, created_at, updated_at
                            FROM pdfdocument
                        """)
                    else:
                        db.execute_sql("""
                            INSERT INTO pdfdocument_new 
                            SELECT id, file_path, file_hash, zotero_key, title, page_count,
                                   last_analyzed, created_at, created_at
                            FROM pdfdocument
                        """)
                    
                    # Drop old table and rename new one
                    db.execute_sql("DROP TABLE pdfdocument")
                    db.execute_sql("ALTER TABLE pdfdocument_new RENAME TO pdfdocument")
                    logger.info("Updated database schema with all required columns")
                    
                except Exception as e:
                    logger.error(f"Error during migration: {str(e)}")
                    raise
            
            try:
                # Drop existing indexes if they exist
                db.execute_sql("DROP INDEX IF EXISTS idx_pdfdocument_file_hash")
                db.execute_sql("DROP INDEX IF EXISTS idx_pdfdocument_zotero_key")
                
                # Create conditional unique indexes
                db.execute_sql("""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_pdfdocument_file_hash 
                    ON pdfdocument(file_hash) 
                    WHERE file_hash IS NOT NULL
                """)
                db.execute_sql("""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_pdfdocument_zotero_key 
                    ON pdfdocument(zotero_key) 
                    WHERE zotero_key IS NOT NULL
                """)
                logger.info("Created/updated database indexes")
            except Exception as e:
                logger.error(f"Error creating indexes: {str(e)}")
                # Continue even if index creation fails
                
        logger.info(f"Database initialized at {db_path}")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

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
        logger.error(f"Error calculating file hash for {file_path}: {str(e)}")
        raise 