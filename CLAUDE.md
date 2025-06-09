# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PaleoDesk is a desktop application suite for paleontologists. The main component is PDFRefinery, a PyQt6-based tool that automatically extracts figures, captions, and metadata from paleontological research papers using machine learning-based layout analysis.

## Core Architecture

- **Technology Stack**: PyQt6 desktop application with SQLite database (Peewee ORM)
- **External Service**: Relies on huridocs Docker service for PDF layout analysis
- **Database**: SQLite with migration system using Peewee ORM

### Key Components

- `PDFRefinery.py` - Main application entry point and window management
- `PDFModels.py` - Database models defining schema for papers, elements, figures, etc.
- `PrComponents.py` - Custom UI components including PDF viewer with overlay rendering
- `PrDialogs.py` - Dialog windows for preferences, element editing, and data entry
- `PDFCommons.py` - Shared utilities, configuration management, and service integration

## Development Commands

```bash
# Run the application
cd pdfrefinery
python PDFRefinery.py

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m unittest discover -s tests

# Run database migrations
python migrate.py

# OCR testing
python ocr_test.py
```

## Database Architecture

The application uses a relational schema with these core entities:
- `Paper` - Research papers with bibliographic metadata
- `Element` - Raw extracted elements from PDF pages
- `Figure` - Processed figure objects with associated elements
- `Caption` - Captions linked to figures
- `Spicule` - Paleontological specimen data

Database migrations are managed through numbered migration files in `migrations/`.

## Key Features

**Implemented:**
- PDF viewer with interactive overlay for detected elements
- Element selection, merging, and figure-caption association
- Zotero integration and file system browsing for paper import
- huridocs service integration for layout analysis
- Database persistence with migration system

**In Development (see tasks/):**
- Undo/redo system for element operations
- Subfigure detection and extraction
- LLM integration for caption parsing and metadata extraction
- Export functionality for extracted data
- Search and filtering capabilities

## Integration Points

- **Zotero**: Direct integration with Zotero library for paper metadata
- **huridocs**: External Docker service for PDF layout analysis
- **File System**: Direct access to PDF files and attachment management

## UI Architecture

The application follows PyQt6 patterns with:
- Custom graphics view for PDF rendering with overlay capabilities
- Signal/slot communication between components  
- Modular dialog system for different data entry workflows
- Event-driven updates for real-time element manipulation