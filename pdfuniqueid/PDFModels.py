from peewee import *
import datetime
import os
import logging
import hashlib
import shutil
from PDFCommons import *
import json

# Initialize logger
logger = logging.getLogger("PDFModels")


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
