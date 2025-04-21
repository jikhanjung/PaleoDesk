import os
import sys
import datetime
import difflib
from pathlib import Path
import importlib.util
from typing import List, Tuple

MIGRATIONS_DIR = Path(__file__).parent / 'migrations'
MODELS_FILE = Path(__file__).parent / 'PDFModels.py'
MODELS_BACKUP = MODELS_FILE.parent / '.models_backup.py'

def get_model_changes() -> List[Tuple[str, str]]:
    """
    Compare current models with backup to detect changes
    Returns list of (change_type, description) tuples
    """
    if not MODELS_BACKUP.exists():
        # First run, just copy current models
        if MODELS_FILE.exists():
            MODELS_BACKUP.write_text(MODELS_FILE.read_text())
        return []
    
    # Read both files
    current = MODELS_FILE.read_text().splitlines()
    backup = MODELS_BACKUP.read_text().splitlines()
    
    changes = []
    in_model = False
    current_model = ""
    
    # Compare files line by line
    for line in difflib.unified_diff(backup, current):
        if line.startswith('class ') and 'Model)' in line:
            current_model = line.split('class ')[1].split('(')[0]
            in_model = True
        elif in_model and line.startswith('+    '):  # New field
            if '=' in line:
                field_name = line.strip().split('=')[0].strip()
                field_type = line.strip().split('=')[1].strip().split('(')[0].strip()
                changes.append(('add', f"Add {field_name} ({field_type}) to {current_model}"))
        elif in_model and line.startswith('-    '):  # Removed field
            if '=' in line:
                field_name = line.strip().split('=')[0].strip()
                changes.append(('remove', f"Remove {field_name} from {current_model}"))
    
    return changes

def suggest_migration_name() -> str:
    """Generate migration name based on model changes"""
    changes = get_model_changes()
    
    if not changes:
        return None
    
    # Generate name based on first change
    change_type, description = changes[0]
    
    if change_type == 'add':
        name = description.lower().replace(' ', '_').replace('(', '').replace(')', '')
    elif change_type == 'remove':
        name = description.lower().replace(' ', '_')
    else:
        name = 'update_models'
    
    return name

def create_migration(name=None):
    """Create a new migration file with the given or generated name"""
    # Generate name if not provided
    if name is None:
        name = suggest_migration_name()
        if name is None:
            print("No model changes detected. Please provide a migration name manually.")
            print("Usage: python migrate.py <migration_name>")
            sys.exit(1)
    
    # Ensure migrations directory exists
    MIGRATIONS_DIR.mkdir(exist_ok=True)
    
    # Get existing migration files
    existing_migrations = [
        f for f in os.listdir(MIGRATIONS_DIR)
        if f.endswith('.py') and f not in ['__init__.py', 'migration_manager.py']
    ]
    
    # Determine next migration number
    if existing_migrations:
        last_num = max(int(f[:3]) for f in existing_migrations)
        next_num = str(last_num + 1).zfill(3)
    else:
        next_num = '000'
    
    # Create migration filename
    filename = f"{next_num}_{name}.py"
    filepath = MIGRATIONS_DIR / filename
    
    # Get detected changes for template
    changes = get_model_changes()
    migration_ops = []
    rollback_ops = []
    
    for change_type, description in changes:
        if change_type == 'add':
            field_name = description.split('Add ')[1].split(' (')[0]
            field_type = description.split('(')[1].split(')')[0]
            table_name = description.split(' to ')[1]
            
            migration_ops.append(
                f"    migrator.add_column('{table_name.lower()}', '{field_name}', "
                f"{field_type}(null=True))"
            )
            rollback_ops.append(
                f"    migrator.drop_column('{table_name.lower()}', '{field_name}')"
            )
    
    # Migration file template
    template = f'''from peewee import *
from playhouse.migrate import SqliteMigrator, migrate as migrate_fields
import datetime

def migrate(db):
    """
    {chr(10).join(changes)}
    """
    migrator = SqliteMigrator(db)
    
    migrate_fields(
{chr(10).join(migration_ops) if migration_ops else "        # TODO: Implement migration operations"}
    )

def rollback(db):
    """
    Rollback the above changes
    """
    migrator = SqliteMigrator(db)
    
    migrate_fields(
{chr(10).join(rollback_ops) if rollback_ops else "        # TODO: Implement rollback operations"}
    )
'''
    
    # Create the file
    with open(filepath, 'w') as f:
        f.write(template)
    
    # Update backup after successful migration creation
    MODELS_FILE.exists() and MODELS_BACKUP.write_text(MODELS_FILE.read_text())
    
    print(f"Created migration file: {filename}")
    print(f"Location: {filepath}")
    print("\nDetected changes:")
    for change in changes:
        print(f"- {change[1]}")
    print("\nRemember to:")
    print("1. Review the generated migration operations")
    print("2. Test both migrate and rollback before deploying")

def main():
    name = None if len(sys.argv) < 2 else sys.argv[1].lower().replace(' ', '_')
    create_migration(name)

if __name__ == "__main__":
    main() 