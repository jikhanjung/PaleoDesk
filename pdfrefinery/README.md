# PDFRefinery

PDFRefinery is a research assistant tool designed for paleontologists. It helps users read PDF research papers and automatically extract useful information, including figures and captions, title, author(s), journal, year, and more. This streamlines the process of gathering and organizing key data from scientific literature, saving researchers time and reducing manual effort. PDFRefinery is valuable because it targets the specific needs of paleontologists, enabling them to efficiently access and utilize critical information from academic papers.

## Core Features
- **Extract Figures and Captions from PDFs:** Automatically identifies and extracts all figures and their associated captions from research paper PDFs.
- **Extract Metadata:** Gathers bibliographic information such as title, authors, journal, and year.
- **Export Extracted Data:** Export all extracted information to formats like CSV, JSON, or Excel.
- **Search and Filter:** Search within and filter the extracted figures, captions, and metadata.
- **User-Friendly Interface:** Organized, editable view for reviewing and correcting extracted data.

## User Experience
- **Target Users:** Paleontologists eager to systematically collect and compile knowledge from research papers.
- **Workflow:**
  1. Retrieve a collection and paper list from a Zotero database.
  2. Select a paper to analyze.
  3. Perform layout analysis using the huridocs PDF layout analysis Docker service.
  4. Review and adjust the extracted layout information within the app.
  5. Save or export the curated data for further use.
- **UI:** Desktop application built with PyQt6 for a native, responsive user experience. Visual display of PDF pages with overlays for detected figures and captions. Easy-to-use tools for editing and confirming extracted information.

## Technical Architecture
- **Input Sources:** Zotero database (direct file access), folders of PDF files, or individual PDF files.
- **PDF Processing:** huridocs PDF layout analysis Docker service (accessed via HTTP requests, returns JSON).
- **Data Storage:** Local SQLite database for papers, figures, captions, and subfigures.
- **UI:** PyQt6-based desktop interface.

## Setup & Requirements
- **Python 3.8+**
- **PyQt6**
- **SQLite**
- **Docker** (for huridocs PDF layout analysis container)
- **Zotero** (for local database access, optional)

### Quick Start
1. Clone this repository.
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure Docker is installed and the huridocs PDF layout analysis container is running locally.
4. Launch the application:
   ```bash
   python PDFRefinery.py
   ```

## Development Roadmap
- **MVP:**
  - Zotero access for retrieving collections and paper lists (via direct DB file access)
  - PDF layout analysis using huridocs Docker container
  - Saving extracted information into a local SQLite database
  - Basic PyQt6 desktop interface
  - Manual adjustment of extracted layout information
- **Next Steps:**
  - Improved UI and workflow
  - Enhanced tools for adjusting layout and extracted data
  - Extraction of subfigures and association with individualized captions
- **Future Enhancements:**
  - Server-based architecture for centralized storage and access
  - Unique paper identification system
  - Cloud/networked database for sharing and collaboration
  - Optional integration with Zotero API
  - Advanced search, filtering, and analytics

## Risks and Mitigations
- **Layout Analysis Accuracy:** huridocs is generally good but not perfect; user correction and future support for other APIs planned.
- **Caption/Subfigure Extraction:** Complex caption styles may require advanced LLMs; fallback to user correction and commercial APIs if needed.
- **LLM Quality/Cost:** Will use local LLMs where possible; commercial APIs as a backup.

## License
[Specify your license here]

## Appendix
- For more details, see [prd.txt](./prd.txt)