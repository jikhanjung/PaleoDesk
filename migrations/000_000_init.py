from peewee import *
from playhouse.migrate import SqliteMigrator, migrate as migrate_fields
import datetime

def migrate(db):
    """
    
    """
    migrator = SqliteMigrator(db)
    
    migrate_fields(
        # TODO: Implement migration operations
    )

def rollback(db):
    """
    Rollback the above changes
    """
    migrator = SqliteMigrator(db)
    
    migrate_fields(
        # TODO: Implement rollback operations
    )
