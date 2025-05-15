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
    element_id = IntegerField()  # ID within the page
    element_type = CharField()  # Type of element (e.g., 'figure', 'table', 'text')
    coordinates = TextField()  # JSON string of coordinates
    content = TextField(null=True)  # Text content or description
    caption = TextField(null=True)  # Associated caption text
    metadata = TextField(null=True)  # JSON string for additional metadata
    image_path = TextField(null=True)  # Path to the image file
    image_binary = BlobField(null=True)  # Binary data of the image
    linked_elements = TextField(null=True)  # JSON string of linked elements
    merged_elements = TextField(null=True)  # JSON string of merged elements
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
            'metadata': json.loads(self.metadata) if self.metadata else {},
            'linked_elements': json.loads(self.linked_elements) if self.linked_elements else [],
            'merged_elements': json.loads(self.merged_elements) if self.merged_elements else [],
            'page_number': self.page_number
        }
        return element_data

class PrFigure(BaseModel):
    """Model for storing figure information"""
    document = ForeignKeyField(PDFDocument, backref='figures')
    figure_number = CharField()
    figure_page_number = IntegerField(null=True)
    part1_prefix = CharField(null=True)
    part1_number = CharField(null=True)
    part2_prefix = CharField(null=True)
    part2_number = CharField(null=True)
    part_separator = CharField(null=True,default='-')
    figure_binary = BlobField(null=True)
    caption_binary = BlobField(null=True)
    caption_text = TextField(null=True)
    parent = ForeignKeyField('self', backref='children', null=True,on_delete="CASCADE")
    created_at = DateTimeField(default=datetime.datetime.now)
    updated_at = DateTimeField(default=datetime.datetime.now)

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