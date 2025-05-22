from PyQt6.QtWidgets import (QWidget, QToolBar, QListWidget, QVBoxLayout, QLabel, 
                           QListWidgetItem, QMessageBox, QHBoxLayout, QPushButton, QScrollArea, QToolTip,
                            QDialog, QButtonGroup, QDialogButtonBox, QRadioButton, QMenu, QApplication,
                            QSplitter, QListView, QTableView, QSizePolicy, QFrame, QHeaderView, QVBoxLayout, QPushButton, QComboBox, QInputDialog )
from PyQt6.QtCore import Qt, QPoint, QPointF, QEvent, pyqtSignal, QRect, QSize, QRectF, pyqtSignal, QTimer, QEvent, QCoreApplication, QAbstractTableModel, QModelIndex, QVariant, QItemSelectionModel
from PyQt6.QtGui import QPixmap, QPainter, QAction, QColor, QPen, QPainterPath, QBrush, QImage, QImageReader, QStandardItem, QStandardItemModel, QCursor
from PDFModels import StructuredElement, PrFigure
from PDFCommons import *
import logging
import json
import datetime
import os
import requests
import sqlite3
import shutil
from peewee import DoesNotExist, chunked
from peewee_migrate import Router
import fitz
from PrDialogs import *
import numpy as np
import cv2
import io
from PIL import Image as PILImage
# Get logger
logger = logging.getLogger('PrComponents')

class PDFViewer(QWidget):
    # Add signal for wheel events
    wheel_scrolled = pyqtSignal(int)  # Signal to emit the scroll amount
    currentPageChanged = pyqtSignal(int)  # Signal to emit the new current page (0-based)
    
    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self.dpr = self.devicePixelRatioF()
        self.pixmap = None
        self.current_page = 0
        self.zoom = 1.0
        self.drag_start = None
        self.drag_pos = None
        self.show_bounding_boxes = True
        self.pdf_document= None
        self.total_pages = 0
        self.initial_load_pages = 3  # Number of pages to load initially

        self.loaded_pages = set()  # Track which pages are loaded
        self.page_pixmaps = {}  # Cache for page pixmaps
        self.bounding_boxes = {}
        self.scaled_pixmaps = {}

        self.page_loading = False  # Flag to prevent multiple simultaneous loads
        self.pan_offset = QPoint(0, 0)  # Add pan offset
        self.last_pan_pos = None  # Add last pan position
        self.current_page_width = 0  # Add current page width
        self.current_page_height = 0  # Add current page height
        self.hovered_box = None  # Track currently hovered bounding box
        self.hovered_box_page = None  # Track page number of hovered box
        self.selected_boxes = []  # Track selected boxes (list of (page_num, box_id) tuples, order preserved, no duplicates)
        # Add these new instance variables
        self.dragging_box = None  # Currently dragged box
        self.drag_start_pos = None  # Mouse position when drag started
        self.original_box_coords = None  # Original coordinates of the box being dragged
        self.resize_edges = None  # Which edges are being resized (e.g., 'n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw')
        self.edge_threshold = 10  # Pixels from edge to trigger resize cursor
        
        # Add element creation variables
        self.creating_element = False
        self.element_start_pos = None
        self.element_current_pos = None
        self.element_type = 'text'  # Default type
        
        # Set up the widget
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # Create zoom buttons
        self.create_zoom_buttons()
        
        # Set minimum size
        self.setMinimumSize(400, 600)
        
        # Set background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), Qt.GlobalColor.white)
        self.setPalette(palette)
    
    def create_zoom_buttons(self):
        """Create zoom control buttons"""
        # Create container widget for buttons
        self.zoom_widget = QWidget(self)
        self.zoom_widget.setObjectName("zoomWidget")
        self.zoom_widget.setStyleSheet("""
            QWidget#zoomWidget {
                background-color: rgba(255, 255, 255, 180);
                border-radius: 5px;
            }
            QPushButton {
                background-color: white;
                border: 1px solid #ccc;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
        """)
        
        # Create layout for buttons
        zoom_layout = QHBoxLayout(self.zoom_widget)
        zoom_layout.setContentsMargins(5, 5, 5, 5)
        zoom_layout.setSpacing(5)
        
        # Create zoom out button
        self.zoom_out_btn = QPushButton("-")
        self.zoom_out_btn.setFixedSize(30, 30)
        self.zoom_out_btn.clicked.connect(lambda: self.set_zoom(self.zoom / 1.1))
        zoom_layout.addWidget(self.zoom_out_btn)
        
        # Create zoom in button
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setFixedSize(30, 30)
        self.zoom_in_btn.clicked.connect(lambda: self.set_zoom(self.zoom * 1.1))
        zoom_layout.addWidget(self.zoom_in_btn)
        
        # Position the zoom widget
        self.zoom_widget.setFixedSize(80, 40)
        self.zoom_widget.move(self.width() - 90, 10)  # Position in top-right corner

    def clear_document(self):
        """Clear the current PDF document and reset state"""
        if self.pdf_document:
            self.pdf_document = None
        self.current_page = 0
        self.loaded_pages.clear()
        self.page_pixmaps.clear()
        self.scaled_pixmaps.clear()
        self.bounding_boxes.clear()
        self.show_bounding_boxes = True
        self.zoom = 1.0
        # Scroll PDF view area to top
        if hasattr(self.main_window, 'pdf_scroll'):
            self.main_window.pdf_scroll.verticalScrollBar().setValue(0)
        self.update()

    def _get_element_info(self, page_num, element_id):
        """Get info for the selected box"""
        #logger.info(f"Getting element info for page {type(page_num)} {page_num}, element {type(element_id)} {element_id}")
        if not self.main_window or not hasattr(self.main_window, 'document_data'):
            return
        
        page_structures = self.main_window.document_data.get('page_structures', {})
        #logger.info(f"page_structures: {page_structures.keys()}")
        page_structure = page_structures.get(str(page_num), {})
        page_elements = page_structure.get('structure', {}).get('elements', [])
        #logger.info(f"page_elements: {page_elements} {len(page_elements)}")
        for element in page_elements:
            #logger.info(f"{type(element)} {element.get('id')} {type(element.get('id'))}")
            if int(element.get('id')) == int(element_id):
                return element

    def set_document(self, pdf_document):
        try:
            """Set the PDF document and reset state"""
            self.clear_document()
            self.pdf_document = pdf_document
            self.figure_count = 0
            self.figure_dict = {}
            self.total_pages = len(self.pdf_document)
            self.load_initial_pages()
            
            # Update the display
            self.update_current_page()
            self.update()
            logger.info(f"Set PDF with {self.total_pages} pages")
            return True
        except Exception as e:
            logger.error(f"Error setting PDF: {str(e)}")
            return False

    def load_initial_pages(self):
        """Load the initial set of pages"""
        try:
            # Load first few pages
            for page_num in range(min(self.initial_load_pages, self.total_pages)):
                self.load_page(page_num)

            default_pixmap_width = self.page_pixmaps[0]['width']
            default_pixmap_height = self.page_pixmaps[0]['height']
            total_height = 0

            for page_num in range(self.total_pages):
                if page_num not in self.page_pixmaps:
                    # Initialize pixmap for unloaded pages
                    self.page_pixmaps[page_num] = {
                        'pixmap': None,
                        'width': default_pixmap_width,
                        'height': default_pixmap_height
                    }
                    total_height += default_pixmap_height
                else:
                    # Update total height for loaded pages
                    total_height += self.page_pixmaps[page_num]['height']
            
            # Start loading next set of pages in background
            #self.load_next_pages()
        except Exception as e:
            logger.error(f"Error loading initial pages: {str(e)}")

    def set_current_page(self, page_num):
        """Set the current page and scroll to it"""
        if self.pdf_document and 0 <= page_num < len(self.pdf_document):
            logger.info(f"[PDFViewer] set_current_page: {page_num}")
            self.current_page = page_num
            #self.display_all_pages()
            self.scroll_to_page(page_num)
            logger.debug(f"Set current page to {page_num + 1}")
            self.currentPageChanged.emit(self.current_page)

    def paintEvent(self, event):
        """Handle painting of the widget (optimized: only paint current_page-1, current_page, current_page+1)"""
        if not self.page_pixmaps:
            return
        
        # Get viewport position
        if not self.main_window or not hasattr(self.main_window, 'pdf_scroll'):
            return

        viewport_height = self.main_window.pdf_scroll.viewport().height()
        scroll_value = self.main_window.pdf_scroll.verticalScrollBar().value()
        viewport_top = scroll_value
        viewport_bottom = scroll_value + viewport_height
        
        logger.debug(f"[paintEvent] Viewport: top={viewport_top}, bottom={viewport_bottom}, current_page={self.current_page}")
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Only paint current_page-1, current_page, current_page+1
        pages_to_paint = [self.current_page - 1, self.current_page, self.current_page + 1]
        current_y = 0
        
        # Calculate total height for all pages to maintain layout
        total_height = 0
        for page_num in sorted(self.page_pixmaps.keys()):
            width = self.page_pixmaps[page_num]['width']
            height = self.page_pixmaps[page_num]['height']
            scale = self.width() / width
            scaled_height = int(height * scale)
            total_height += scaled_height
            
        # Set widget size to match total height
        self.setMinimumHeight(total_height)
        
        # First pass: Paint pages and bounding boxes, and collect element centers
        element_centers = {}  # (page_num, element_id) -> (center_x, center_y)
        page_y_offsets = {}   # page_num -> current_y
        for page_num in sorted(self.page_pixmaps.keys()):
            # Always set the y offset for every page
            page_y_offsets[page_num] = current_y
            if page_num not in pages_to_paint:
                width = self.page_pixmaps[page_num]['width']
                height = self.page_pixmaps[page_num]['height']
                scale = self.width() / width
                scaled_height = int(height * scale)
                current_y += scaled_height
                continue
            # Ensure the page is loaded
            if self.page_pixmaps[page_num]['pixmap'] is None:
                self.load_page(page_num)
                if page_num not in self.page_pixmaps:
                    continue
            pixmap = self.page_pixmaps[page_num]['pixmap']
            width = self.page_pixmaps[page_num]['width']
            height = self.page_pixmaps[page_num]['height']
            scale = self.width() / width
            scaled_height = int(height * scale)
            page_top = current_y
            target_rect = QRect(0, int(current_y), self.width(), scaled_height)
            painter.drawPixmap(target_rect, pixmap)
            if self.show_bounding_boxes and self.bounding_boxes:
                page_boxes = self.bounding_boxes.get(page_num, [])
                for element in page_boxes:
                    if 'coordinates' in element and len(element['coordinates']) == 4:
                        coords = element['coordinates']
                        x1 = int(coords[0]['x'] * self.width())
                        y1 = int(coords[0]['y'] * scaled_height + current_y)
                        x2 = int(coords[2]['x'] * self.width())
                        y2 = int(coords[2]['y'] * scaled_height + current_y)
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        box_id = element.get('id')
                        element_centers[(page_num, box_id)] = (center_x, center_y)
                        category = element.get('category', 'text').lower() 
                        category_text = category+ " " + str(int(page_num)+1) + "-" + str(int(element.get('id', ''))+1)
                        
                        # Check if box is selected
                        box_id = element.get('id')
                        is_selected = (page_num, box_id) in self.selected_boxes
                        
                        # Get color based on selection state
                        color = ELEMENT_COLORS.get(category, QColor(255, 0, 0, 64))
                        if is_selected:
                            # Make selected boxes more opaque and slightly brighter
                            color.setAlpha(128)
                            color = color.lighter(120)
                        else:
                            color.setAlpha(64)
                        if element.get('merged_elements'):
                            color = QColor(128,128,0,196)
                        if element.get('linked_elements'):
                            color = QColor(128,0,0,196)
                        if len(coords) == 4:
                            x1 = int(coords[0]['x'] * self.width())
                            y1 = int(coords[0]['y'] * scaled_height + current_y)
                            x2 = int(coords[2]['x'] * self.width())
                            y2 = int(coords[2]['y'] * scaled_height + current_y)
                            category = element.get('category', 'text').lower() 
                            category_text = category+ " " + str(int(page_num)+1) + "-" + str(int(element.get('id', ''))+1)
                            
                            # Check if box is selected
                            box_id = element.get('id')
                            is_selected = (page_num, box_id) in self.selected_boxes
                            
                            # Get color based on selection state
                            color = ELEMENT_COLORS.get(category, QColor(255, 0, 0, 64))
                            if is_selected:
                                # Make selected boxes more opaque and slightly brighter
                                color.setAlpha(128)
                                color = color.lighter(120)
                            else:
                                color.setAlpha(64)
                            logger.debug(f"Drawing bounding box for {category_text} at {x1}, {y1}, {x2}, {y2}, color: {color.getRgb()}")
                            
                            pen = QPen(color, 2)
                            painter.setPen(pen)
                            painter.setBrush(color)
                            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                            
                            # Draw label with appropriate color
                            label_color = color
                            label_color.setAlpha(128)
                            painter.setPen(Qt.GlobalColor.black)
                            painter.setBrush(label_color)
                            label_rect = painter.boundingRect(x1, y1 - 20, 100, 20, 
                                                            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                                                            category_text)
                            painter.drawRect(label_rect)
                            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, category_text)
            current_y += scaled_height
            
        # Second pass: Draw S-shaped curves for merged elements
        line_pen = QPen(QColor(200, 180, 0, 200), 2, Qt.PenStyle.DashLine)
        painter.setPen(line_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)  # Ensure no fill for connecting lines
        for (page_num, box_id), (center_x, center_y) in element_centers.items():
            element = self._get_element_info(page_num, box_id)
            merged = element.get('merged_elements') if element else None
            if element and merged and len(merged) > 1:
                # Draw S-shaped curves for each consecutive pair
                for i in range(len(merged) - 1):
                    prev_page, prev_id = merged[i]
                    next_page, next_id = merged[i + 1]
                    #logger.info(f"Drawing S-shaped curve for {prev_page}, {prev_id} -> {next_page}, {next_id}")
                    prev_element = self._get_element_info(prev_page, prev_id)
                    next_element = self._get_element_info(next_page, next_id)
                    if prev_element and next_element and 'coordinates' in prev_element and 'coordinates' in next_element:
                        # Get bottom center of prev_element
                        prev_coords = prev_element['coordinates']
                        prev_x1 = int(prev_coords[0]['x'] * self.width())
                        prev_x2 = int(prev_coords[2]['x'] * self.width())
                        prev_y2 = int(prev_coords[2]['y'] * self.page_pixmaps[int(prev_page)]['height'] * (self.width() / self.page_pixmaps[int(prev_page)]['width']) + page_y_offsets.get(int(prev_page), 0))
                        prev_bottom = QPointF((prev_x1 + prev_x2) / 2, prev_y2)
                        # Get top center of next_element
                        next_coords = next_element['coordinates']
                        next_x1 = int(next_coords[0]['x'] * self.width())
                        next_x2 = int(next_coords[2]['x'] * self.width())
                        next_y1 = int(next_coords[0]['y'] * self.page_pixmaps[int(next_page)]['height'] * (self.width() / self.page_pixmaps[int(next_page)]['width']) + page_y_offsets.get(int(next_page), 0))
                        next_top = QPointF((next_x1 + next_x2) / 2, next_y1)
                        # Control points for S-curve
                        dy = next_top.y() - prev_bottom.y()
                        dx = next_top.x() - prev_bottom.x()
                        vertical_offset = 200
                        ctrl1 = QPointF(prev_bottom.x(), prev_bottom.y() + vertical_offset)
                        ctrl2 = QPointF(next_top.x(), next_top.y() - vertical_offset)
                        path = QPainterPath()
                        path.moveTo(prev_bottom)
                        path.cubicTo(ctrl1, ctrl2, next_top)
                        painter.drawPath(path)

        # Draw element creation rectangle if in progress
        if self.creating_element and self.element_start_pos is not None and self.element_current_pos is not None:
            painter.setPen(QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.DashLine))
            x1 = min(self.element_start_pos.x(), self.element_current_pos.x())
            y1 = min(self.element_start_pos.y(), self.element_current_pos.y())
            x2 = max(self.element_start_pos.x(), self.element_current_pos.x())
            y2 = max(self.element_start_pos.y(), self.element_current_pos.y())
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

        # --- Convex hull utility (Andrew's monotone chain) ---
        def convex_hull(points):
            # points: list of (x, y)
            points = sorted(set(points))
            if len(points) <= 1:
                return points
            def cross(o, a, b):
                return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
            lower = []
            for p in points:
                while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                    lower.pop()
                lower.append(p)
            upper = []
            for p in reversed(points):
                while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                    upper.pop()
                upper.append(p)
            return lower[:-1] + upper[:-1]

        # --- Grouping for linked_elements ---
        # Map: frozenset of (page_num, box_id) -> set of all corner points
        linked_groups = {}
        element_to_group = {}
        for (page_num, box_id), (center_x, center_y) in element_centers.items():
            element = self._get_element_info(page_num, box_id)
            if element and element.get('linked_elements'):
                group = set()
                for ref in element['linked_elements']:
                    ref_page, ref_id = ref
                    group.add((int(ref_page), ref_id))
                group.add((page_num, box_id))
                group_key = frozenset(group)
                if group_key not in linked_groups:
                    linked_groups[group_key] = set()
                element_to_group[(page_num, box_id)] = group_key
        # Collect all corner points for each group
        for group_key in linked_groups:
            for page_num, box_id in group_key:
                element = self._get_element_info(page_num, box_id)
                if element and 'coordinates' in element and len(element['coordinates']) == 4:
                    coords = element['coordinates']
                    # Find the page's y offset and scale
                    width = self.page_pixmaps[page_num]['width']
                    height = self.page_pixmaps[page_num]['height']
                    scale = self.width() / width
                    scaled_height = int(height * scale)
                    y_offset = page_y_offsets.get(page_num, 0)
                    for c in coords:
                        x = int(c['x'] * self.width())
                        y = int(c['y'] * scaled_height + y_offset)
                        linked_groups[group_key].add((x, y))
        # Draw convex hulls with internal padding
        hull_pen = QPen(QColor(0, 128, 255, 180), 2, Qt.PenStyle.SolidLine)
        hull_brush = QBrush(QColor(0, 128, 255, 60))
        expand_factor = 1.05  # 1.1 means 10% outward padding
        for group_key, points in linked_groups.items():
            if len(points) >= 3:
                hull = convex_hull(list(points))
                # Compute centroid
                cx = sum(x for x, y in hull) / len(hull)
                cy = sum(y for x, y in hull) / len(hull)
                padded_hull = []
                for x, y in hull:
                    px = cx + (x - cx) * expand_factor
                    py = cy + (y - cy) * expand_factor
                    padded_hull.append(QPoint(int(px), int(py)))
                painter.setPen(hull_pen)
                painter.setBrush(hull_brush)
                painter.drawPolygon(*padded_hull)

    def set_bounding_boxes(self, boxes):
        """Set bounding boxes for all pages"""
        if not boxes:
            return
            
        # Clear existing boxes
        self.bounding_boxes = {}
        
        # Get the main window instance
        if not self.main_window or not hasattr(self.main_window, 'document_data'):
            return
            
        # Get all page structures
        page_structures = self.main_window.document_data.get('page_structures', {})
        
        # Set boxes for each page
        for page_num, structure in page_structures.items():
            page_boxes = structure.get('structure', {}).get('elements', [])
            if page_boxes:
                self.bounding_boxes[int(page_num)] = page_boxes
                #logger.debug(f"Set {len(page_boxes)} bounding boxes for page {int(page_num) + 1}")
        
        self.update()

    def reset_page(self):
        """Reset the current page's boxes to their initial state"""
        if not self.main_window or not hasattr(self.main_window, 'document_data'):
            return
            
        # Get initial page structure for current page
        initial_structure = self.main_window.document_data.get('initial_page_structures', {}).get(str(self.current_page))
        if not initial_structure:
            logger.warning(f"No initial structure found for page {self.current_page}")
            return
            
        # Get current page structure
        current_structure = self.main_window.document_data.get('page_structures', {}).get(str(self.current_page))
        if not current_structure:
            logger.warning(f"No current structure found for page {self.current_page}")
            return
            
        # Update current page structure with initial elements
        current_structure['structure']['elements'] = initial_structure['structure']['elements'].copy()
        
        # Update bounding boxes
        self.bounding_boxes[self.current_page] = initial_structure['structure']['elements']
        
        # Update structured content view
        #if hasattr(self.main_window, 'structured_view'):
        #    self.main_window.structured_view.update_content(self.main_window.document_data['page_structures'])
        
        self.update()
        logger.info(f"Reset page {self.current_page} to initial state")

    def toggle_bounding_boxes(self, show):
        """Toggle bounding box display"""
        self.show_bounding_boxes = show
        self.update()

    def set_zoom(self, new_zoom):
        """Set the zoom level"""
        if 0.1 <= new_zoom <= 5.0:  # Limit zoom range
            self.zoom = new_zoom
            # Clear page cache when zoom changes
            self.page_pixmaps.clear()
            self.loaded_pages.clear()
            # Reload current page with new zoom
            self.update_current_page()
            self.update()
        
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if self.creating_element and event.button() == Qt.MouseButton.LeftButton:
            self.element_start_pos = event.pos()
            self.element_current_pos = event.pos()
            return
            
        if event.button() == Qt.MouseButton.LeftButton or event.button() == Qt.MouseButton.RightButton:
            # Check if we're over a box
            pos = event.pos()
            box_info = self._check_bounding_box_hover(pos)
            
            if box_info:
                page_num, box = box_info
                box_id = box.get('id')
                extra_select = []
                # Gather merged/linked elements if present
                if 'merged_elements' in box and box['merged_elements']:
                    for ref in box['merged_elements']:
                        if ref not in extra_select:
                            extra_select.append((int(ref[0]), ref[1]))
                if 'linked_elements' in box and box['linked_elements']:
                    for ref in box['linked_elements']:
                        if ref not in extra_select:
                            extra_select.append((int(ref[0]), ref[1]))
                logger.debug(f"extra_select: {extra_select}, {box['linked_elements'] if 'linked_elements' in box else ''} {box['merged_elements'] if 'merged_elements' in box else '' }")
                # Always include the clicked box itself
                if (page_num, box_id) not in extra_select:
                    extra_select.append((page_num, box_id))
                
                if event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier):
                    # Toggle selection with Ctrl
                    for sel in extra_select:
                        if sel in self.selected_boxes:
                            self.selected_boxes.remove(sel)
                        else:
                            if sel not in self.selected_boxes:
                                self.selected_boxes.append(sel)
                else:
                    # Single selection without Ctrl
                    if not all(sel in self.selected_boxes for sel in extra_select):
                        self.selected_boxes = []
                        for sel in extra_select:
                            if sel not in self.selected_boxes:
                                self.selected_boxes.append(sel)
                
                # If we're over a box, start dragging or resizing
                if self.dragging_box is not None:
                    self.drag_start_pos = event.pos()
                    # Store original coordinates and element id
                    self.original_box_coords = self.dragging_box['coordinates'].copy()
                    self.dragging_box_id = self.dragging_box.get('id')  # Store the element id
                else:
                    # Normal panning behavior
                    self.last_pan_pos = event.pos()
            else:
                # Clicked outside any box, clear selection
                self.selected_boxes = []
                # Normal panning behavior
                self.last_pan_pos = event.pos()
            logger.debug(f"selected_boxes: {self.selected_boxes}")
            self.update()  # Update to show selection changes

    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
        if self.creating_element and self.element_start_pos is not None:
            self.element_current_pos = event.pos()
            self.update()  # Trigger repaint to show the rectangle
            return
            
        if not self.main_window:
            return

        if self.drag_start_pos and self.dragging_box and event.buttons() & Qt.MouseButton.LeftButton:
            # Calculate the movement delta in viewport coordinates
            delta_x = event.pos().x() - self.drag_start_pos.x()
            delta_y = event.pos().y() - self.drag_start_pos.y()
            
            # Convert delta to PDF coordinates
            widget_width = self.width()
            first_page_width = next((self.page_pixmaps[i]['width'] for i in range(len(self.page_pixmaps)) if i in self.page_pixmaps), None)
            if not first_page_width:
                return
                
            width_ratio = widget_width / (first_page_width * self.zoom)
            
            # Get page dimensions
            page_width = self.page_pixmaps[self.current_box_page]['width']
            page_height = self.page_pixmaps[self.current_box_page]['height']
            
            # Convert movement to normalized coordinates
            screen_page_width = page_width * self.zoom * width_ratio
            screen_page_height = page_height * self.zoom * width_ratio
            
            delta_x_norm = delta_x / screen_page_width
            delta_y_norm = delta_y / screen_page_height

            if self.resize_edges:
                # Resizing - update only the edges being dragged
                coords = self.dragging_box['coordinates']
                if 'n' in self.resize_edges:
                    coords[0]['y'] = self.original_box_coords[0]['y'] + delta_y_norm
                    coords[1]['y'] = self.original_box_coords[1]['y'] + delta_y_norm
                if 's' in self.resize_edges:
                    coords[2]['y'] = self.original_box_coords[2]['y'] + delta_y_norm
                    coords[3]['y'] = self.original_box_coords[3]['y'] + delta_y_norm
                if 'w' in self.resize_edges:
                    coords[0]['x'] = self.original_box_coords[0]['x'] + delta_x_norm
                    coords[3]['x'] = self.original_box_coords[3]['x'] + delta_x_norm
                if 'e' in self.resize_edges:
                    coords[1]['x'] = self.original_box_coords[1]['x'] + delta_x_norm
                    coords[2]['x'] = self.original_box_coords[2]['x'] + delta_x_norm
            else:
                # Normal dragging - move all points
                for i in range(4):
                    self.dragging_box['coordinates'][i]['x'] = self.original_box_coords[i]['x'] + delta_x_norm
                    self.dragging_box['coordinates'][i]['y'] = self.original_box_coords[i]['y'] + delta_y_norm
            
            # Update drag start position to current position for smooth continuous dragging
            self.drag_start_pos = event.pos()
            
            self.update()  # Redraw the view
        elif self.last_pan_pos and event.buttons() & Qt.MouseButton.LeftButton:
            # Normal panning behavior
            delta = event.pos() - self.last_pan_pos
            self.pan_offset += delta
            self.last_pan_pos = event.pos()
            self.update()
        else:
            # Check if mouse is over a bounding box
            self._check_bounding_box_hover(event.pos())

    def mouseReleaseEvent(self, event):
        #logger.info(f"Mouse release event: {event}, {event.button()}")
        """Handle mouse release events"""
        if self.creating_element and event.button() == Qt.MouseButton.LeftButton and self.element_start_pos is not None:
            # Get the main window instance
            if not self.main_window or not hasattr(self.main_window, 'document_data'):
                logger.error("No main window or document data found")
                return
                
            # Calculate the rectangle coordinates
            x1 = min(self.element_start_pos.x(), self.element_current_pos.x())
            y1 = min(self.element_start_pos.y(), self.element_current_pos.y())
            x2 = max(self.element_start_pos.x(), self.element_current_pos.x())
            y2 = max(self.element_start_pos.y(), self.element_current_pos.y())
            
            # Get current page height from page_pixmaps
            if self.current_box_page in self.page_pixmaps:
                page_height = self.page_pixmaps[self.current_box_page]['height']
                
                # Calculate total height of pages before current page
                total_height_before = 0
                for i in range(self.current_cursor_page):
                    if i in self.page_pixmaps:
                        total_height_before += self.page_pixmaps[i]['height']
                
                # Calculate scaling to fit width while maintaining aspect ratio
                scale = self.width() / self.page_pixmaps[self.current_box_page]['width']
                scaled_height = int(page_height * scale)  # Convert to int
                
                # Adjust y coordinates by subtracting the height of previous pages and accounting for scaling
                y1_adjusted = (y1 - total_height_before) / (scale * self.zoom)
                y2_adjusted = (y2 - total_height_before) / (scale * self.zoom)
                
                # Convert screen coordinates to relative coordinates
                rel_x1 = x1 / (self.width() * self.zoom)
                rel_y1 = y1_adjusted / page_height
                rel_x2 = x2 / (self.width() * self.zoom)
                rel_y2 = y2_adjusted / page_height
                
                logger.info(f"Creating new element on page {self.current_box_page}")
                logger.info(f"Screen coordinates: ({x1}, {y1}) to ({x2}, {y2})")
                logger.info(f"Page height: {page_height}")
                logger.info(f"Total height before current page: {total_height_before}")
                logger.info(f"Scale: {scale}, Zoom: {self.zoom}")
                logger.info(f"Adjusted y coordinates: ({y1_adjusted}, {y2_adjusted})")
                logger.info(f"Relative coordinates: ({rel_x1:.3f}, {rel_y1:.3f}) to ({rel_x2:.3f}, {rel_y2:.3f})")
                
                # Get current page elements to determine new element ID
                page_key = str(self.current_box_page)
                if page_key in self.main_window.document_data['page_structures']:
                    page_structure = self.main_window.document_data['page_structures'][page_key]
                    if 'structure' in page_structure and 'elements' in page_structure['structure']:
                        elements = page_structure['structure']['elements']
                        # show elements' page, id and category
                        logger.info(f"Elements on current page: {[{'page': page_key, 'id': e['id'], 'category': e['category']} for e in elements]}")
                        
                        new_id = int(len(elements))  # Use element count as ID
                        logger.info(f"Current page has {len(elements)} elements, new element ID will be {new_id}")
                        
                        # Create new element with normalized coordinates
                        new_element = {
                            'id': new_id,
                            'category': self.element_type,
                            'coordinates': [
                                {'x': rel_x1, 'y': rel_y1},  # top-left
                                {'x': rel_x2, 'y': rel_y1},  # top-right
                                {'x': rel_x2, 'y': rel_y2},  # bottom-right
                                {'x': rel_x1, 'y': rel_y2}   # bottom-left
                            ],
                            'content': {'text': ''},
                            'attributes': {
                                'page_width': self.current_page_width,
                                'page_height': self.current_page_height
                            }
                        }
                        logger.info(f"Created new element: {new_element}")
                        
                        # Add element to current page
                        elements.append(new_element)
                        logger.info(f"Added element to page structure, now has {len(elements)} elements")
                        
                        # Save the new element to database
                        self.save_element(self.current_box_page, new_id)
                        
                        # Update bounding boxes
                        if self.current_box_page not in self.bounding_boxes:
                            self.bounding_boxes[str(self.current_box_page)] = []
                        #self.bounding_boxes[self.current_cursor_page].append(new_element)
                        #logger.info(f"Added element to bounding boxes for page {self.current_cursor_page}")
                        
                        # Update structured content view
                        #if hasattr(main_window, 'structured_view'):
                        #    logger.info("Updating structured content view")
                        #    main_window.structured_view.update_content(main_window.document_data['page_structures'])
                        #    logger.info("Saving session")
                        #    #main_window.save_session()  # Auto-save after adding element
                        #else:
                        #    logger.warning("No structured view found in main window")
                else:
                    logger.error(f"Page {page_key} not found in page structures")
            else:
                logger.error(f"Page {self.current_box_page} not found in page_pixmaps")
            
            # Reset element creation state
            self.creating_element = False
            self.element_start_pos = None
            self.element_current_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        #logger.info(f"Mouse release event: {event}, {event.button()}, selected_boxes: {self.selected_boxes}")
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_pan_pos = None
            if self.dragging_box:
                logger.debug(f"Box release - Page: {self.current_box_page}, Box ID: {self.dragging_box_id}")
                logger.debug(f"New coordinates: {self.dragging_box['coordinates']}")

                # Update the page structure with new box coordinates
                if self.current_box_page is not None and self.current_box_page in self.bounding_boxes:
                    # Find and update the box in the page structure
                    page_boxes = self.bounding_boxes[self.current_box_page]
                    for i, box in enumerate(page_boxes):
                        if box is self.dragging_box:  # Compare by reference
                            # Update the box coordinates in the page structure
                            page_boxes[i] = self.dragging_box
                            logger.debug(f"Updated box in bounding_boxes[{self.current_box_page}][{i}]")
                            break
                    
                    # Update the main window's page structure if available
                    # Find the main window by traversing up the widget hierarchy
                    if hasattr(self.main_window, 'document_data'):
                        logger.debug("Found document_data in main window")
                        if 'page_structures' in self.main_window.document_data:
                            logger.debug(f"Found page_structures in document_data")
                            page_key = str(self.current_box_page)
                            if page_key in self.main_window.document_data['page_structures']:
                                logger.debug(f"Found page {page_key} in page_structures")
                                page_structure = self.main_window.document_data['page_structures'][page_key]
                                if 'structure' in page_structure and 'elements' in page_structure['structure']:
                                    elements = page_structure['structure']['elements']
                                    logger.debug(f"Found {len(elements)} elements in page structure")
                                    # Find and update the corresponding element by id
                                    for element in elements:
                                        if element.get('id') == self.dragging_box_id:
                                            logger.debug(f"Found matching element with id {self.dragging_box_id}")
                                            # Update coordinates
                                            element['coordinates'] = self.dragging_box['coordinates']
                                            
                                            # Save the updated element to database
                                            self.save_element(self.current_box_page, self.dragging_box_id)
                                            
                                            # Update the structured content view
                                            #if hasattr(main_window, 'structured_view'):
                                            #    logger.debug("Updating structured content view")
                                            #    main_window.structured_view.update_content(main_window.document_data['page_structures'])
                                            #    #main_window.save_session()
                                            #    logger.debug("Structured content view update called")
                                            #else:
                                            #    logger.warning("No structured_view found in main window")
                                            break
                                    else:
                                        logger.warning(f"No element found with id {self.dragging_box_id}")
                                else:
                                    logger.warning("No 'structure' or 'elements' in page_structure")
                            else:
                                logger.warning(f"Page {page_key} not found in page_structures")
                        else:
                            logger.warning("No page_structures found in document_data")
                    else:
                        logger.warning("No document_data found in main window")
                
                # Reset dragging state
                self.drag_start_pos = None
                self.original_box_coords = None
                self.dragging_box_id = None
                self.resize_edges = None
                # Keep dragging_box set so we maintain hover state
                
                # Force a redraw
                self.update()

    def save_element(self, page_num, element_id):
        """Save the element to database and update the page structure"""
        # Get the element from the page structure
        if not self.main_window or not hasattr(self.main_window, 'document_data'):
            return
        page_key = str(page_num)
        page_structure = self.main_window.document_data['page_structures'][page_key]
        element = page_structure['structure']['elements'][int(element_id)]
        #logger.info(f"Saving element {element} from page {page_num}")

        document = self.main_window.document_record
        if not document:
            logger.error("No document found in database")
            return

        try:
            # Create or update the structured element
            structured_element = StructuredElement.get_or_create(
                document=document,
                page_number=page_num,
                element_id=int(element_id),
                defaults={
                    'element_type': element.get('category', 'unknown'),
                    'figure_number': element.get('figure_number', 0),
                    'coordinates': json.dumps(element.get('coordinates', [])),
                    'content': json.dumps(element.get('content', {})),
                    'caption': json.dumps(element.get('caption', {})),
                    'metadata': json.dumps(element.get('metadata', {})),
                    'linked_elements': json.dumps(element.get('linked_elements', [])),
                    'merged_elements': json.dumps(element.get('merged_elements', [])),
                    'created_at': datetime.datetime.now(),
                    'updated_at': datetime.datetime.now()
                }
            )[0]

            old_type = structured_element.element_type
            new_type = element.get('category', 'unknown')
            
            # Update the element if it already existed
            structured_element.element_type = element.get('category', 'unknown')
            structured_element.figure_number = element.get('figure_number', 0)
            structured_element.coordinates = json.dumps(element.get('coordinates', []))
            structured_element.content = json.dumps(element.get('content', {}))
            structured_element.caption = json.dumps(element.get('caption', {}))
            structured_element.metadata = json.dumps(element.get('metadata', {}))
            structured_element.linked_elements = json.dumps(element.get('linked_elements', []))
            structured_element.merged_elements = json.dumps(element.get('merged_elements', []))
            structured_element.updated_at = datetime.datetime.now()
            structured_element.save()

            #logger.info(f"Saved row_id {structured_element.id} for element {element_id} from page {page_num} to database {old_type} -> {new_type}")
            
        except Exception as e:
            logger.error(f"Error saving element to database: {str(e)}")
            QMessageBox.warning(self, "Save Error", 
                f"Error saving element to database: {str(e)}")

    def update_current_page(self, scroll_value=None):
        """Update current page based on scroll position or current state"""
        if not self.pdf_document:
            return
            
        # Get the main window instance
        if not self.main_window or not hasattr(self.main_window, 'pdf_scroll'):
            return
            
        # Get viewport height
        viewport_height = self.main_window.pdf_scroll.viewport().height()
        
        # If scroll_value is provided, use it to calculate viewport center
        if scroll_value is not None:
            viewport_center = scroll_value + viewport_height / 2
        else:
            # Otherwise, get current scroll position
            scroll_value = self.main_window.pdf_scroll.verticalScrollBar().value()
            viewport_center = scroll_value + viewport_height / 2
            
        logger.debug(f"[update_current_page] Viewport center: {viewport_center}, current_page: {self.current_page}, scroll_value: {scroll_value}")
        
        # If bounding boxes are enabled, ensure all pages are loaded
        buffer = [ self.current_page - 1, self.current_page, self.current_page + 1 ]
        for page_num in buffer:
            if page_num not in self.loaded_pages and page_num >= 0 and page_num < len(self.pdf_document):
                self.load_page(page_num)

        if False:
            if self.show_bounding_boxes and self.bounding_boxes:
                for page_num in range(len(self.pdf_document)):
                    if page_num not in self.loaded_pages:
                        self.load_page(page_num)
        
        # Find the page that contains the viewport center
        current_height = 0
        for i, page_data in self.page_pixmaps.items():
            # Calculate scaled height considering zoom rate
            width = page_data['width']
            height = page_data['height']
            scale = self.width() / width
            scaled_height = int(height * scale)
            
            page_bottom = current_height + scaled_height
            logger.debug(f"[update_current_page] Page {i} range: {current_height} to {page_bottom}, scale={scale}, zoom={self.zoom}")
            
            if current_height <= viewport_center < page_bottom:
                if self.current_page != i:
                    #logger.info(f"[PDFViewer] Changing current_page from {self.current_page} to {i} (emitting currentPageChanged)")
                    self.current_page = i
                    self.currentPageChanged.emit(self.current_page)
                break
            current_height = page_bottom
            
        # Update the display
        self.update()

    def resizeEvent(self, event):
        """Handle widget resize"""
        super().resizeEvent(event)
        logger.debug("[PDFViewer::resizeEvent] Widget resize event triggered")
        # Reposition zoom widget
        if hasattr(self, 'zoom_widget'):
            self.zoom_widget.move(self.width() - 90, 10)
            
        # If we have pages, update their display without reloading
        if self.page_pixmaps:
            logger.debug(f"[resizeEvent] Updating current page display")
            # Update current page display
            self.update_current_page()
            self.update()
    
    def sizeHint(self):
        """Return the size hint for the widget"""
        if not self.bounding_boxes:
            return super().sizeHint()
            
        # Calculate total height based on loaded pages
        total_height = sum(page_data['height'] for page_data in self.page_pixmaps.values())
        return QSize(self.width(), total_height)
        
    def minimumSizeHint(self):
        """Return the minimum size hint for the widget"""
        if not self.bounding_boxes:
            return super().minimumSizeHint()
            
        # Calculate total height based on loaded pages
        total_height = sum(page_data['height'] for page_data in self.page_pixmaps.values())
        return QSize(self.width(), total_height)
    
    def scroll_to_page(self, page_num):
        """Scroll to make the specified page visible"""
        if not self.bounding_boxes or page_num < 0 or page_num >= len(self.bounding_boxes):
            return
            
        # Calculate the y position for the target page
        y_position = 0
        for i in range(page_num):
            y_position += self.bounding_boxes[i]['structure']['structure']['elements'][0]['attributes']['page_height']
            
        # Center the page in the viewport
        viewport_height = self.height()
        target_y = int(y_position - (viewport_height - self.bounding_boxes[page_num]['structure']['structure']['elements'][0]['attributes']['page_height']) / 2)
        
        # Update pan offset
        self.pan_offset.setY(-target_y)
        self.update()
        logger.debug(f"Scrolled to page {page_num + 1} at y={target_y}")

    def set_current_page(self, page_num):
        """Set the current page and scroll to it"""
        if self.pdf_document and 0 <= page_num < len(self.pdf_document):
            logger.info(f"[PDFViewer] set_current_page: {page_num}")
            self.current_page = page_num
            self.scroll_to_page(page_num)
            logger.debug(f"Set current page to {page_num + 1}")
            self.currentPageChanged.emit(self.current_page)

    def load_next_pages(self):
        return
        """Load next set of pages in background"""
        if self.page_loading or not self.pdf_document:
            return
            
        self.page_loading = True
        try:
            # Find the highest loaded page number
            max_loaded = max(self.loaded_pages) if self.loaded_pages else -1
            
            # Load next set of pages
            next_pages = range(max_loaded + 1, min(max_loaded + self.initial_load_pages + 1, len(self.pdf_document)))
            for page_num in next_pages:
                self.load_page(page_num)
            
            # Update display if current page was loaded
            if self.current_page in self.loaded_pages:
                self.update_current_page()
                self.update()
                
            # Update total height
            total_height = sum(page_data['height'] for page_data in self.page_pixmaps.values())
            self.setMinimumHeight(total_height)
            self.updateGeometry()
            self.adjustSize()
            self.update()
            
            # If there are more pages to load, schedule next batch
            if max_loaded + self.initial_load_pages < len(self.pdf_document):
                QTimer.singleShot(100, self.load_next_pages)
        finally:
            self.page_loading = False

    def load_page(self, page_num):
        """Load a single page and cache its pixmap"""
        logger.debug(f"[PDFViewer] load_page: {page_num}")
        if page_num in self.loaded_pages or page_num >= len(self.pdf_document):
            return
        
        try:
            page = self.pdf_document[page_num]
            # Use a higher zoom factor for better quality
            display_zoom = self.zoom
            render_zoom = display_zoom * 2  # Double the zoom for rendering
            matrix = fitz.Matrix(render_zoom, render_zoom)
            #logger.info(f"Page {page_num + 1} - render_zoom: {render_zoom}, display_zoom: {display_zoom}")
            pix = page.get_pixmap(matrix=matrix)
            #logger.info(f"Page {page_num + 1} - pixmap size: {pix.width}, {pix.height}")
            
            # Convert to QImage
            img = QImage(pix.samples, pix.width, pix.height, 
                       pix.stride, QImage.Format.Format_RGB888)
            
            # Create QPixmap
            pixmap = QPixmap.fromImage(img)
            
            # Scale the pixmap to the display size
            display_width = int(self.width() * display_zoom)  # Convert to integer
            # Calculate height maintaining aspect ratio
            aspect_ratio = pixmap.height() / pixmap.width()
            display_height = int(display_width * aspect_ratio)
            
            #pixmap = pixmap.scaled(display_width, display_height,
            #                     Qt.AspectRatioMode.KeepAspectRatio,
            #                     Qt.TransformationMode.SmoothTransformation)
            # show pixmap size and display size
            #logger.debug(f"Pixmap size: {pixmap.size()}, display size: {display_width}, {display_height}")
            
            # Store page data
            self.page_pixmaps[page_num] = {
                'pixmap': pixmap,
                'width': pixmap.width(),
                'height': pixmap.height()
            }
            self.loaded_pages.add(page_num)
            logger.debug(f"Loaded page {page_num + 1} - width: {pixmap.width()}, height: {pixmap.height()}")
        except Exception as e:
            logger.error(f"Error loading page {page_num}: {str(e)}")

    def leaveEvent(self, event):
        """Handle mouse leave events"""
        if self.hovered_box is not None:
            self.hovered_box = None
            QToolTip.hideText()
            logger.debug("Mouse left widget, hiding tooltip")

    def _check_bounding_box_hover(self, pos):
        """Check if mouse is over a bounding box and return (page_num, box) if found"""
        if not self.bounding_boxes:
            return None
            
        # Get scroll position for PDF coordinate calculation only
        if not self.main_window or not hasattr(self.main_window, 'pdf_scroll'):
            return None
            
        scroll_value = self.main_window.pdf_scroll.verticalScrollBar().value()
        
        # Calculate widget to pixmap ratio for proper scaling
        widget_width = self.width()
        first_page_width = next((self.page_pixmaps[i]['width'] for i in range(len(self.page_pixmaps)) if i in self.page_pixmaps), None)
        if not first_page_width:
            return None
            
        # This ratio accounts for how much the PDF is scaled to fit the widget width
        width_ratio = widget_width / (first_page_width * self.zoom)
        
        # Use raw mouse position for viewport coordinates
        viewport_x = pos.x()
        viewport_y = pos.y()
        
        # Find current page based on viewport position
        current_y = 0
        current_page = None

        # Use viewport_y directly for page detection since it's already in screen coordinates
        for page_num in range(len(self.page_pixmaps)):
            if page_num in self.page_pixmaps:
                page_height = self.page_pixmaps[page_num]['height']  # Use actual height, not scaled
                scaled_height = page_height * self.zoom * width_ratio  # Scale height by both zoom and ratio
                if current_y <= viewport_y < current_y + scaled_height:
                    current_page = page_num
                    break
                current_y += scaled_height

        if current_page is None:
            self.dragging_box = None
            self.current_box_page = None
            self.resize_edges = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return None

        # Check boxes on current page
        page_boxes = self.bounding_boxes.get(current_page, [])
        for box in page_boxes:
            if 'coordinates' not in box:
                continue
                
            coords = box['coordinates']
            if len(coords) != 4:
                continue
                
            # Get coordinates from the box
            x1 = coords[0]['x']
            y1 = coords[0]['y']
            x2 = coords[2]['x']  # Use point 2 for bottom-right
            y2 = coords[2]['y']
            
            # Get page dimensions for scaling
            page_width = self.page_pixmaps[current_page]['width']
            page_height = self.page_pixmaps[current_page]['height']
            
            # Convert normalized coordinates to screen space, applying both zoom and width ratio
            screen_x1 = x1 * page_width * self.zoom * width_ratio
            screen_x2 = x2 * page_width * self.zoom * width_ratio
            
            # Calculate screen y coordinates relative to viewport, applying both zoom and width ratio
            screen_y1 = y1 * page_height * self.zoom * width_ratio + current_y
            screen_y2 = y2 * page_height * self.zoom * width_ratio + current_y
            
            # Check if mouse is within box boundaries using viewport coordinates
            if (screen_x1 - self.edge_threshold <= viewport_x <= screen_x2 + self.edge_threshold and 
                screen_y1 - self.edge_threshold <= viewport_y <= screen_y2 + self.edge_threshold):
                
                self.dragging_box = box
                self.current_box_page = current_page
                
                # Only enable resize for selected box (not just single selection)
                box_id = box.get('id')
                is_selected = (current_page, box_id) in self.selected_boxes
                #is_only_selected = len(self.selected_boxes) == 1 and is_selected
                if is_selected:
                    # Determine which edges the cursor is near
                    near_left = abs(viewport_x - screen_x1) <= self.edge_threshold
                    near_right = abs(viewport_x - screen_x2) <= self.edge_threshold
                    near_top = abs(viewport_y - screen_y1) <= self.edge_threshold
                    near_bottom = abs(viewport_y - screen_y2) <= self.edge_threshold
                    # Set resize edges and cursor based on position
                    if near_top and near_left:
                        self.resize_edges = 'nw'
                        self.setCursor(Qt.CursorShape.SizeFDiagCursor)
                    elif near_top and near_right:
                        self.resize_edges = 'ne'
                        self.setCursor(Qt.CursorShape.SizeBDiagCursor)
                    elif near_bottom and near_left:
                        self.resize_edges = 'sw'
                        self.setCursor(Qt.CursorShape.SizeBDiagCursor)
                    elif near_bottom and near_right:
                        self.resize_edges = 'se'
                        self.setCursor(Qt.CursorShape.SizeFDiagCursor)
                    elif near_left:
                        self.resize_edges = 'w'
                        self.setCursor(Qt.CursorShape.SizeHorCursor)
                    elif near_right:
                        self.resize_edges = 'e'
                        self.setCursor(Qt.CursorShape.SizeHorCursor)
                    elif near_top:
                        self.resize_edges = 'n'
                        self.setCursor(Qt.CursorShape.SizeVerCursor)
                    elif near_bottom:
                        self.resize_edges = 's'
                        self.setCursor(Qt.CursorShape.SizeVerCursor)
                    else:
                        self.resize_edges = None
                        self.setCursor(Qt.CursorShape.OpenHandCursor if not self.drag_start_pos else Qt.CursorShape.ClosedHandCursor)
                else:
                    self.resize_edges = None
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                
                return (current_page, box)
                
        self.dragging_box = None
        self.current_box_page = None
        self.resize_edges = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        return None

    def start_element_creation(self):
        """Start the element creation process"""
        self.creating_element = True
        self.setCursor(Qt.CursorShape.CrossCursor)
        # Show element type selection dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Element Type")
        layout = QVBoxLayout()
        
        # Create radio buttons for element types
        button_group = QButtonGroup()
        for element_type in ELEMENT_TYPES:
            radio = QRadioButton(element_type.capitalize())
            radio.setChecked(element_type == 'text')
            button_group.addButton(radio)
            layout.addWidget(radio)
        
        # Add OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Get selected type
            for button in button_group.buttons():
                if button.isChecked():
                    self.element_type = button.text().lower()
                    break
        else:
            self.creating_element = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for selection"""
        if event.key() == Qt.Key.Key_A and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # Ctrl+A: Select all boxes on current page
            if self.current_page in self.bounding_boxes:
                for box in self.bounding_boxes[self.current_page]:
                    if 'id' in box:
                        if (self.current_page, box['id']) not in self.selected_boxes:
                            self.selected_boxes.append((self.current_page, box['id']))
                self.update()
        elif event.key() == Qt.Key.Key_Escape:
            # Escape: Clear all selections
            self.selected_boxes = []
            self.update()
        elif event.key() == Qt.Key.Key_Delete:
            # Delete: Remove selected boxes
            if self.selected_boxes:
                for page_num, box_id in list(self.selected_boxes):
                    if page_num in self.bounding_boxes:
                        self.bounding_boxes[page_num] = [b for b in self.bounding_boxes[page_num] 
                                                       if b.get('id') != box_id]
                self.selected_boxes = []
                self.update()
        else:
            super().keyPressEvent(event)

    def contextMenuEvent(self, event):
        """Show context menu for boxes"""
        # Check if we're over a box
        result = self._check_bounding_box_hover(event.pos())
        if not result:
            return
            
        page_num, box = result
        if not box or 'id' not in box:
            return
            
        # Only show menu if the box is selected
        if (page_num, box['id']) not in self.selected_boxes:
            return

        box_type = set()
        for box in self.selected_boxes:
            box_info = self._get_element_info(box[0], box[1])
            category = box_info.get('category', 'text').lower() 
            box_type.add(category)

        menu = QMenu(self)

        # show_info
        show_info_action = menu.addAction("Show Info")
        show_info_action.triggered.connect(self._show_info)

        # Determine all_linked and all_merged for selected boxes
        all_linked = True
        all_merged = True
        any_linked = False
        any_merged = False
        for sel in self.selected_boxes:
            box_info = self._get_element_info(sel[0], sel[1])
            is_linked = box_info and 'linked_elements' in box_info and box_info['linked_elements'] and len(box_info['linked_elements']) > 0
            is_merged = box_info and 'merged_elements' in box_info and box_info['merged_elements'] and len(box_info['merged_elements']) > 0
            if not is_linked:
                all_linked = False
            if not is_merged:
                all_merged = False
            if is_linked:
                any_linked = True
            if is_merged:
                any_merged = True

        # merge/link actions
        if len(self.selected_boxes) > 1:
            if len(box_type) == 1:
                if not all_merged:
                    merge_action = menu.addAction("Merge")
                    merge_action.triggered.connect(lambda: self._merge_selected_boxes())
            else:
                if not all_linked:
                    link_action = menu.addAction("Link")
                    link_action.triggered.connect(lambda: self._link_selected_boxes())

        # Unlink/Unmerge actions for all selected
        if len(self.selected_boxes) > 0:
            if all_linked:
                unlink_action = menu.addAction("Unlink")
                unlink_action.triggered.connect(lambda: self._unlink_selected_boxes())
            if all_merged:
                unmerge_action = menu.addAction("Unmerge")
                unmerge_action.triggered.connect(lambda: self._unmerge_selected_boxes())

        menu.addSeparator()

        # Add "Change Block Type" as an enabled menu item without action
        change_type_action = menu.addAction("Change Block Type")
        change_type_action.triggered.connect(lambda: None)  # Connect to empty lambda
        if len(box_type) > 1:
            change_type_action.setEnabled(False)
        else:
            change_type_action.setEnabled(True)

        # Disable type change if any selected element is linked
        disable_type_change = False
        for sel in self.selected_boxes:
            box_info = self._get_element_info(sel[0], sel[1])
            if box_info and 'linked_elements' in box_info and box_info['linked_elements'] and len(box_info['linked_elements']) > 0:
                disable_type_change = True
                break
        if disable_type_change:
            change_type_action.setEnabled(False)

        # Add element type actions with indentation
        for element_type in ELEMENT_TYPES:
            if element_type == 'figure':
                # Add a submenu for figure subtypes
                figure_menu = QMenu("    figure (subtype)", menu)
                for fig_type in FIGURE_SUBTYPES:
                    fig_action = figure_menu.addAction(f"        {fig_type}")
                    fig_action.triggered.connect(lambda checked, t=fig_type: self._change_selected_boxes_type(t))
                    if disable_type_change:
                        fig_action.setEnabled(False)
                menu.addMenu(figure_menu)
            else:
                action = menu.addAction(f"    {element_type}")  # 4 spaces for indentation
                action.triggered.connect(lambda checked, t=element_type: self._change_selected_boxes_type(t))
                if disable_type_change:
                    action.setEnabled(False)
        
        menu.addSeparator()
        
        # Add delete action
        delete_action = menu.addAction("Delete")
        delete_action.triggered.connect(self._delete_selected_boxes)
        
        # Show menu
        menu.exec(event.globalPos())

    def _show_info(self):
        """Show info for the selected box"""
        #if len(self.selected_boxes) != 1:
        #    return

        element_data = []
        for page_num, element_id in self.selected_boxes:
            logger.info(f"page_num: {page_num} element_id: {element_id}")
            element_data.append(self._get_element_info(page_num, element_id))

        dialog = ElementInfoDialog(element_data, self.main_window)
        dialog.exec()

    def _change_selected_boxes_type(self, new_type):
        """Change type for all selected boxes"""
        # Get the main window instance
        if not self.main_window or not hasattr(self.main_window, 'document_data'):
            return
            
        logger.debug(f"Changing type to {new_type} for {len(self.selected_boxes)} boxes")
        
        document = self.main_window.document_record
        if not document:
            logger.error("No document found in database")
            return

        try:
            for page_num, box_id in self.selected_boxes:
                if page_num in self.bounding_boxes:
                    for box in self.bounding_boxes[page_num]:
                        if box.get('id') == box_id:
                            # Update UI
                            box['category'] = new_type
                            logger.debug(f"Changed type of box {box_id} on page {page_num} to {new_type}")
                            
                            # Update document_data
                            page_key = int(page_num)
                            if page_key in self.main_window.document_data['page_structures']:
                                page_structure = self.main_window.document_data['page_structures'][page_key]
                                if 'structure' in page_structure and 'elements' in page_structure['structure']:
                                    elements = page_structure['structure']['elements']
                                    for element in elements:
                                        if element.get('id') == box_id:
                                            element['category'] = new_type
                                            break
                            
                            # Update database if element exists
                            try:
                                element = StructuredElement.get(
                                    (StructuredElement.document == document) &
                                    (StructuredElement.page_number == page_num) &
                                    (StructuredElement.element_id == box_id)
                                )
                                element.element_type = new_type
                                element.updated_at = datetime.datetime.now()
                                element.save()
                                logger.debug(f"Updated database for element {box_id} on page {page_num} to type {new_type}")
                            except DoesNotExist:
                                logger.debug(f"No database entry found for element {box_id} on page {page_num}")
                                continue
            
            # Update UI
            self.update()
            
            # Update structured content view
            #if hasattr(self.main_window, 'structured_view'):
            #    self.main_window.structured_view.update_content(self.main_window.document_data['page_structures'])
                
            # Save session to ensure changes are persisted
            #self.main_window.save_session()
                
        except Exception as e:
            logger.error(f"Error updating element type in database: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            QMessageBox.warning(self, "Update Error", 
                f"Error updating element type in database: {str(e)}")

    def _merge_selected_boxes(self):
        """Merge all selected boxes"""
        document = self.main_window.document_record
        if not document:
            logger.error("No document found in database")
            return

        element_list = []
        for elem in self.selected_boxes:
            #box_info = self._get_element_info(box[0], box[1])
            element_list.append([elem[0],elem[1]])

        # sort element_list by page number and int(element_id)
        element_list.sort(key=lambda x: (int(x[0]), int(x[1])))
        logger.info(f"element_list: {element_list}")

        for elem in element_list:
            elem_info = self._get_element_info(elem[0], elem[1])
            element = StructuredElement.get(
                (StructuredElement.document == document) &
                (StructuredElement.page_number == elem[0]) &
                (StructuredElement.element_id == elem[1])
            )
            element.merged_elements = json.dumps(element_list)
            #logger.info(f"element: {element.merged_elements}")
            element.save()
            elem_info['merged_elements'] = element_list
            #logger.info(f"Merged elements {element_list} on page {elem[0]}")
        self.selected_boxes = []
        self.update()

    def _unmerge_selected_boxes(self):
        """Unmerge all selected boxes"""
        document = self.main_window.document_record
        if not document:
            logger.error("No document found in database")
            return
        
        for box in self.selected_boxes:
            box_info = self._get_element_info(box[0], box[1])
            merged_elements = box_info['merged_elements']
            for merged_element in merged_elements:
                elem_info = self._get_element_info(merged_element[0], merged_element[1])
                elem_row_info = StructuredElement.get(  
                    (StructuredElement.document == document) &
                    (StructuredElement.page_number == merged_element[0]) &
                    (StructuredElement.element_id == merged_element[1])
                )
                elem_row_info.merged_elements = None
                elem_row_info.save()
                elem_info['merged_elements'] = []
        self.selected_boxes = []
        self.update()

    def _link_selected_boxes(self):
        """Link all selected boxes"""
        document = self.main_window.document_record
        if not document:
            logger.error("No document found in database")
            return

        element_list = []
        for box in self.selected_boxes:
            #box_info = self._get_element_info(box[0], box[1])
            element_list.append([box[0],box[1]])
        
        # sort element_list by page number and int(element_id)
        element_list.sort(key=lambda x: (int(x[0]), int(x[1])))

        for box in self.selected_boxes:
            elem_info = self._get_element_info(box[0], box[1])
            elem_row_info = StructuredElement.get(
                (StructuredElement.document == document) &
                (StructuredElement.page_number == box[0]) &
                (StructuredElement.element_id == box[1])
            )
            elem_row_info.linked_elements = json.dumps(element_list)
            elem_row_info.save()
            elem_info['linked_elements'] = element_list
            logger.debug(f"Linked elements {element_list} on page {box[0]}")
        self.selected_boxes = []
        self.update()

    def _unlink_selected_boxes(self):
        """Unlink all selected boxes"""
        document = self.main_window.document_record
        if not document:
            logger.error("No document found in database")
            return
        
        for box in self.selected_boxes:
            box_info = self._get_element_info(box[0], box[1])
            linked_elements = box_info['linked_elements']
            for linked_element in linked_elements:
                elem_info = self._get_element_info(linked_element[0], linked_element[1])
                logger.debug(f"linked_element: {linked_element} {elem_info}")
                elem_row_info = StructuredElement.get(
                    (StructuredElement.document == document) &
                    (StructuredElement.page_number == linked_element[0]) &
                    (StructuredElement.element_id == linked_element[1])
                )
                elem_row_info.linked_elements = None
                elem_row_info.save()
                elem_info['linked_elements'] = []
        self.selected_boxes = []
        self.update()

    def _delete_selected_boxes(self):
        """Delete all selected boxes"""
            
        document = self.main_window.document_record
        if not document:
            logger.error("No document found in database")
            return

        try:
            # Process each selected box
            for page_num, box_id in list(self.selected_boxes):
                # Remove the element from document_data
                page_key = str(page_num)
                if page_key in self.main_window.document_data['page_structures']:
                    page_structure = self.main_window.document_data['page_structures'][page_key]
                    if 'structure' in page_structure and 'elements' in page_structure['structure']:
                        elements = page_structure['structure']['elements']
                        deleted_index = None
                        
                        # Find and remove the element
                        for i, element in enumerate(elements):
                            if element.get('id') == box_id:
                                deleted_index = i
                                elements.pop(i)
                                logger.debug(f"Deleted element {box_id} from page {page_num}")
                                break
                        
                        if deleted_index is not None:
                            # Adjust element IDs for remaining elements
                            for i in range(deleted_index, len(elements)):
                                elements[i]['id'] = int(i)
                                logger.debug(f"Adjusted element ID from {i+1} to {i}")
                            
                            # Delete the element from StructuredElement table
                            StructuredElement.delete().where(
                                (StructuredElement.document == document) &
                                (StructuredElement.page_number == page_num) &
                                (StructuredElement.element_id == box_id)
                            ).execute()
                            
                            # Update remaining elements in StructuredElement table
                            for i in range(deleted_index, len(elements)):
                                element = elements[i]
                                StructuredElement.update(
                                    element_id=int(i)
                                ).where(
                                    (StructuredElement.document == document) &
                                    (StructuredElement.page_number == page_num) &
                                    (StructuredElement.element_id == int(i+1))
                                ).execute()
                
                # Remove from UI
                if page_num in self.bounding_boxes:
                    self.bounding_boxes[page_num] = [b for b in self.bounding_boxes[page_num] 
                                                   if b.get('id') != box_id]
            
            # Clear selection
            self.selected_boxes = []
            
            # Update UI
            self.update()
            
            # Update structured content view
            #if hasattr(self.main_window, 'structured_view'):
            #    self.main_window.structured_view.update_content(self.main_window.document_data['page_structures'])
            
            # Save session to update session data
            #self.main_window.save_session()
            logger.info("Updated database after element deletion")
                
        except Exception as e:
            logger.error(f"Error updating database after element deletion: {str(e)}")
            QMessageBox.warning(self, "Database Error", 
                f"Error updating database after element deletion: {str(e)}")

    def _extract_caption_text(self):
        """Extract text from caption image using OCR"""
        if not self.selected_boxes:
            return
            
        # Get the first selected box
        page_num, box_id = self.selected_boxes[0]
        element = self._get_element_info(page_num, box_id)
        if not element:
            return
            
        # Find caption element - either the selected element itself or a linked caption
        caption_element = None
        if element.get('category', '').lower() == 'caption':
            caption_element = element
        elif element.get('linked_elements'):
            for linked_page, linked_id in element['linked_elements']:
                linked_element = self._get_element_info(linked_page, linked_id)
                if linked_element and linked_element.get('category', '').lower() == 'caption':
                    caption_element = linked_element
                    break
                    
        if not caption_element:
            return
            
        # Check if caption already has text
        if caption_element.get('content', {}).get('text'):
            QMessageBox.information(self, "Caption Text", 
                "Caption already has text content.")
            return
            
        try:
            # Get caption coordinates
            coords = caption_element.get('coordinates', [])
            if not coords or len(coords) < 4:
                return
                
            # Get page pixmap
            if page_num not in self.page_pixmaps:
                self.load_page(page_num)
            if page_num not in self.page_pixmaps:
                return
                
            page_pixmap = self.page_pixmaps[page_num]['pixmap']
            if not page_pixmap:
                return
            # wait cursor
            QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
                
            # Convert coordinates to absolute pixels
            page_width = self.page_pixmaps[page_num]['width']
            page_height = self.page_pixmaps[page_num]['height']
            
            x1 = int(coords[0]['x'] * page_width)
            y1 = int(coords[0]['y'] * page_height)
            x2 = int(coords[2]['x'] * page_width)
            y2 = int(coords[2]['y'] * page_height)
            
            # Extract caption region
            caption_pixmap = page_pixmap.copy(x1, y1, x2 - x1, y2 - y1)
            
            caption_text = extract_caption_text(caption_pixmap)
                            
            if caption_text:
                # Update caption element with extracted text
                caption_element['content'] = {'text': caption_text.strip()}
                
                # Update database
                self.save_element(caption_element.get('page', page_num), caption_element.get('id'))
                
                # Update UI
                self.update()
                
                QMessageBox.information(self, "Caption Extraction", 
                    f"Successfully extracted caption text:\n\n{caption_text.strip()}")
            else:
                QMessageBox.warning(self, "Caption Extraction", 
                    "No text could be extracted from the caption image.")
                
            # restore cursor
            QApplication.restoreOverrideCursor()
                    
        except Exception as e:
            # restore cursor
            QApplication.restoreOverrideCursor()
            logger.error(f"Error extracting caption text: {str(e)}")
            QMessageBox.warning(self, "Caption Extraction Error", 
                f"Error extracting caption text: {str(e)}")

    def contextMenuEvent(self, event):
        """Show context menu for boxes"""
        # Check if we're over a box
        result = self._check_bounding_box_hover(event.pos())
        if not result:
            return
            
        page_num, box = result
        if not box or 'id' not in box:
            return
            
        # Only show menu if the box is selected
        if (page_num, box['id']) not in self.selected_boxes:
            return

        box_type = set()
        for box in self.selected_boxes:
            box_info = self._get_element_info(box[0], box[1])
            category = box_info.get('category', 'text').lower() 
            box_type.add(category)

        menu = QMenu(self)

        # show_info
        show_info_action = menu.addAction("Show Info")
        show_info_action.triggered.connect(self._show_info)

        # Check if we should show Extract Caption action
        should_show_extract = False
        if len(self.selected_boxes) == 1:  # Only for single selection
            selected_page, selected_id = self.selected_boxes[0]
            selected_element = self._get_element_info(selected_page, selected_id)
            
            if selected_element:
                # Check if selected element is a caption without text
                if selected_element.get('category', '').lower() == 'caption':
                    if not selected_element.get('content', {}).get('text'):
                        should_show_extract = True
                # Or if it's linked to a caption without text
                elif selected_element.get('linked_elements'):
                    for linked_page, linked_id in selected_element['linked_elements']:
                        linked_element = self._get_element_info(linked_page, linked_id)
                        if (linked_element and 
                            linked_element.get('category', '').lower() == 'caption' and 
                            not linked_element.get('content', {}).get('text')):
                            should_show_extract = True
                            break

        if should_show_extract:
            extract_action = menu.addAction("Extract Caption")
            extract_action.triggered.connect(self._extract_caption_text)
            menu.addSeparator()

        # Determine all_linked and all_merged for selected boxes
        all_linked = True
        all_merged = True
        any_linked = False
        any_merged = False
        for sel in self.selected_boxes:
            box_info = self._get_element_info(sel[0], sel[1])
            is_linked = box_info and 'linked_elements' in box_info and box_info['linked_elements'] and len(box_info['linked_elements']) > 0
            is_merged = box_info and 'merged_elements' in box_info and box_info['merged_elements'] and len(box_info['merged_elements']) > 0
            if not is_linked:
                all_linked = False
            if not is_merged:
                all_merged = False
            if is_linked:
                any_linked = True
            if is_merged:
                any_merged = True

        # merge/link actions
        if len(self.selected_boxes) > 1:
            if len(box_type) == 1:
                if not all_merged:
                    merge_action = menu.addAction("Merge")
                    merge_action.triggered.connect(lambda: self._merge_selected_boxes())
            else:
                if not all_linked:
                    link_action = menu.addAction("Link")
                    link_action.triggered.connect(lambda: self._link_selected_boxes())

        # Unlink/Unmerge actions for all selected
        if len(self.selected_boxes) > 0:
            if all_linked:
                unlink_action = menu.addAction("Unlink")
                unlink_action.triggered.connect(lambda: self._unlink_selected_boxes())
            if all_merged:
                unmerge_action = menu.addAction("Unmerge")
                unmerge_action.triggered.connect(lambda: self._unmerge_selected_boxes())

        menu.addSeparator()

        # Add "Change Block Type" as an enabled menu item without action
        change_type_action = menu.addAction("Change Block Type")
        change_type_action.triggered.connect(lambda: None)  # Connect to empty lambda
        if len(box_type) > 1:
            change_type_action.setEnabled(False)
        else:
            change_type_action.setEnabled(True)

        # Disable type change if any selected element is linked
        disable_type_change = False
        for sel in self.selected_boxes:
            box_info = self._get_element_info(sel[0], sel[1])
            if box_info and 'linked_elements' in box_info and box_info['linked_elements'] and len(box_info['linked_elements']) > 0:
                disable_type_change = True
                break
        if disable_type_change:
            change_type_action.setEnabled(False)

        # Add element type actions with indentation
        for element_type in ELEMENT_TYPES:
            if element_type == 'figure':
                # Add a submenu for figure subtypes
                figure_menu = QMenu("    figure (subtype)", menu)
                for fig_type in FIGURE_SUBTYPES:
                    fig_action = figure_menu.addAction(f"        {fig_type}")
                    fig_action.triggered.connect(lambda checked, t=fig_type: self._change_selected_boxes_type(t))
                    if disable_type_change:
                        fig_action.setEnabled(False)
                menu.addMenu(figure_menu)
            else:
                action = menu.addAction(f"    {element_type}")  # 4 spaces for indentation
                action.triggered.connect(lambda checked, t=element_type: self._change_selected_boxes_type(t))
                if disable_type_change:
                    action.setEnabled(False)
        
        menu.addSeparator()
        
        # Add delete action
        delete_action = menu.addAction("Delete")
        delete_action.triggered.connect(self._delete_selected_boxes)
        
        # Show menu
        menu.exec(event.globalPos())

class StructuredContentView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_doc = None  # Store current PDF document
        self.pixmap_cache = {}  # Cache for item pixmaps
        self.main_window = parent
        logger.debug("StructuredContentView initialized")
        self.init_ui()
        
        # Install event filter for smooth scrolling
        self.content_list_widget.viewport().installEventFilter(self)
        
    def eventFilter(self, obj, event):
        """Handle wheel events for smooth scrolling"""
        if obj == self.content_list_widget.viewport() and event.type() == QEvent.Type.Wheel:
            # Reduce scroll speed by adjusting delta
            scroll_factor = 0.4  # Adjust this value to control scroll speed (smaller = slower)
            delta = event.angleDelta().y()
            reduced_delta = int(delta * scroll_factor)
            
            # Scroll the viewport directly
            scrollbar = self.content_list_widget.verticalScrollBar()
            scrollbar.setValue(scrollbar.value() - reduced_delta)
            return True
            
        return super().eventFilter(obj, event)

    def init_ui(self):
        """Initialize the UI"""
        # Create main layout
        layout = QVBoxLayout()
        
        # Create toolbar for view mode switching
        toolbar = QToolBar()
        toolbar.setMovable(False)
        
        # Create view mode actions
        self.list_view_action = QAction("List View", self)
        self.list_view_action.setCheckable(True)
        self.list_view_action.setChecked(False)
        self.list_view_action.triggered.connect(lambda: self.switch_view_mode('list'))
        
        self.icon_view_action = QAction("Icon View", self)
        self.icon_view_action.setCheckable(True)
        self.icon_view_action.setChecked(True)  # Set icon view as default
        self.icon_view_action.triggered.connect(lambda: self.switch_view_mode('icon'))
        
        # Add actions to toolbar
        toolbar.addAction(self.list_view_action)
        toolbar.addAction(self.icon_view_action)
        
        # Add Save to Database button
        self.save_action = QAction("Save to Database", self)
        self.save_action.setEnabled(False)  # Disabled for now until we implement the feature
        self.save_action.setToolTip("This feature will be implemented in a future update")
        toolbar.addAction(self.save_action)
        
        layout.addWidget(toolbar)
        
        # Create list widget for both views
        self.content_list_widget = QListWidget()
        self.content_list_widget.setViewMode(QListWidget.ViewMode.IconMode)  # Set icon mode as default
        self.content_list_widget.setIconSize(QSize(200, 200))
        self.content_list_widget.setSpacing(10)
        self.content_list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.content_list_widget.setWrapping(False)  # Single column
        self.content_list_widget.setFlow(QListWidget.Flow.TopToBottom)  # Vertical flow
        
        # Enable smooth pixel-based scrolling
        self.content_list_widget.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.content_list_widget.setHorizontalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        
        self.content_list_widget.itemClicked.connect(self._handle_item_click)
        self.content_list_widget.itemDoubleClicked.connect(self._show_element_info)
        
        layout.addWidget(self.content_list_widget)
        self.setLayout(layout)
        
        # Initialize with icon view
        self.current_view_mode = 'icon'
        self.switch_view_mode('icon')
        
    def _get_cached_pixmap(self, item):
        """Get or create a cached pixmap for an item"""
        # Create a unique cache key using page number, id, and coordinates
        coords = item.get('coordinates', [])
        coord_str = ''
        if coords and len(coords) >= 4:
            coord_str = f"_{coords[0]['x']}_{coords[0]['y']}_{coords[2]['x']}_{coords[2]['y']}"
        cache_key = f"{item['page_number']}_{item.get('id', '')}{coord_str}"
        #logger.info(f"cache_key: {cache_key}")
        
        if cache_key in self.pixmap_cache:
            return self.pixmap_cache[cache_key]
            
        try:
            page = self.current_doc[item['page_number'] ]  # Convert to 0-based index
            zoom = 2  # Reduced zoom for better performance
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix)
            
            # Convert to QImage
            img = QImage(pix.samples, pix.width, pix.height, 
                       pix.stride, QImage.Format.Format_RGB888)
            
            # Get coordinates
            if coords and len(coords) >= 4:
                # Convert relative coordinates to absolute
                x1 = int(coords[0]['x'] * pix.width)
                y1 = int(coords[0]['y'] * pix.height)
                x2 = int(coords[2]['x'] * pix.width)
                y2 = int(coords[2]['y'] * pix.height)
                
                # Clip the region
                clipped_img = img.copy(x1, y1, x2 - x1, y2 - y1)
                
                # Create QPixmap and scale to reasonable size
                pixmap = QPixmap.fromImage(clipped_img)
                max_size = 400
                pixmap = pixmap.scaled(max_size, max_size, 
                                     Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
                
                # Cache the pixmap
                self.pixmap_cache[cache_key] = pixmap
                return pixmap
        except Exception as e:
            logger.error(f"Error creating image for item on page {item['page_number']} {item['id']}: {str(e)}")
        
        return None

    def _handle_item_click(self, item):
        """Handle single click on an item to scroll to the corresponding page"""
        # Get the main window instance
        if not self.main_window:
            return
            
        # Get the widget associated with the item
        widget = self.content_list_widget.itemWidget(item)
        if not widget or not hasattr(widget, 'element_data'):
            return
            
        # Get the page number from the element data
        page_num = widget.element_data.get('page')
        logger.debug(f"Page number: {page_num}")
        if page_num is not None:
            # Update current page in PDF viewer
            self.main_window.current_page = page_num - 1  # Convert to 0-based index
            self.main_window.current_page_input.setText(str(page_num))
            self.main_window.pdf_viewer.current_page = page_num - 1
            self.main_window.go_to_page()
            self.main_window.update_navigation()
        
    def set_document(self, doc):
        """Set the current PDF document"""
        self.current_doc = doc
        
    def switch_view_mode(self, mode):
        """Switch between list and icon views"""
        if mode == 'list':
            self.content_list_widget.setViewMode(QListWidget.ViewMode.ListMode)
            self.content_list_widget.setIconSize(QSize(32, 32))  # Smaller icons for list view
            self.content_list_widget.setSpacing(2)  # Less spacing for list view
            self.list_view_action.setChecked(True)
            self.icon_view_action.setChecked(False)
        else:
            self.content_list_widget.setViewMode(QListWidget.ViewMode.IconMode)
            self.content_list_widget.setIconSize(QSize(200, 200))  # Larger icons for icon view
            self.content_list_widget.setSpacing(10)  # More spacing for icon view
            self.list_view_action.setChecked(False)
            self.icon_view_action.setChecked(True)
        
        # Update content in the current view
        if hasattr(self, '_current_content'):
            self.update_content(self._current_content)
        
    def _find_nearest_caption(self, element, page_elements):
        """Find the nearest caption for an element on the same page"""
        element_type = element.get('category', '').lower()
        element_coords = element.get('coordinates', [])
        
        if not element_coords or len(element_coords) < 4:
            return None
            
        # Get element's bounding box
        element_x1 = element_coords[0]['x']
        element_y1 = element_coords[0]['y']
        element_x2 = element_coords[2]['x']
        element_y2 = element_coords[2]['y']
        element_center_x = (element_x1 + element_x2) / 2
        element_center_y = (element_y1 + element_y2) / 2
        
        # Find all captions on the page
        captions = []
        for other_element in page_elements:
            if other_element.get('category', '').lower() == 'caption':
                caption_coords = other_element.get('coordinates', [])
                if caption_coords and len(caption_coords) >= 4:
                    caption_x1 = caption_coords[0]['x']
                    caption_y1 = caption_coords[0]['y']
                    caption_x2 = caption_coords[2]['x']
                    caption_y2 = caption_coords[2]['y']
                    caption_center_x = (caption_x1 + caption_x2) / 2
                    caption_center_y = (caption_y1 + caption_y2) / 2
                    
                    # Calculate distance between element and caption
                    dx = caption_center_x - element_center_x
                    dy = caption_center_y - element_center_y
                    distance = (dx * dx + dy * dy) ** 0.5
                    
                    captions.append({
                        'element': other_element,
                        'distance': distance,
                        'is_below': caption_y1 > element_y2,  # Caption is below the element
                        'is_above': caption_y2 < element_y1,  # Caption is above the element
                        'is_left': caption_x2 < element_x1,   # Caption is to the left
                        'is_right': caption_x1 > element_x2   # Caption is to the right
                    })
        
        if not captions:
            return None
            
        # Sort captions by distance
        captions.sort(key=lambda x: x['distance'])
        
        # For tables and figures, prefer captions below
        if element_type in ['table', 'figure']:
            below_captions = [c for c in captions if c['is_below']]
            if below_captions:
                return below_captions[0]['element']
        
        # For images and pictures, prefer captions below or to the right
        elif element_type in ['image', 'picture']:
            below_or_right = [c for c in captions if c['is_below'] or c['is_right']]
            if below_or_right:
                return below_or_right[0]['element']
        
        # If no preferred position found, return the closest caption
        return captions[0]['element']

    def _sort_elements_by_position(self, items):
        """Sort elements based on page number and position in a two-column layout"""
        def get_position_key(item):
            # First sort by page number
            page = item.get('page', 0)
            
            coords = item.get('coordinates', [])
            if not coords or len(coords) < 4:
                return (page, 0, 0)  # Default position if coordinates are missing
                
            # Get center point of the element
            x1 = coords[0]['x']
            y1 = coords[0]['y']
            x2 = coords[2]['x']
            y2 = coords[2]['y']
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Determine column (left or right)
            # Assuming page is divided into two equal columns
            is_left_column = center_x < 0.5
            
            # Determine vertical position (top or bottom)
            # Using 0.5 as the midpoint
            is_top = center_y < 0.5
            
            # Return a tuple for sorting:
            # First element: page number
            # Second element: 0 for left column, 1 for right column
            # Third element: 0 for top, 1 for bottom
            return (page, 0 if is_left_column else 1, 0 if is_top else 1)
        
        # Sort items based on their position
        return sorted(items, key=get_position_key)

    def _get_category_from_caption(self, caption):
        """Extract category from caption text (e.g., 'Figure 1. ...' -> 'figure')"""
        if not caption:
            return None
            
        # Split by whitespace and get first word
        first_word = caption.split()[0].lower()
        
        # Map common category words
        category_map = {
            'figure': 'figure',
            'fig': 'figure',
            'table': 'table',
            'image': 'image',
            'picture': 'picture',
            'photo': 'picture',
            'diagram': 'figure',
            'chart': 'figure',
            'graph': 'figure'
        }
        
        return category_map.get(first_word)

    def _get_element_info(self, page_num, element_id):
        """Get info for the selected box"""
        logger.info(f"Getting element info for page {type(page_num)} {page_num}, element {type(element_id)} {element_id}")
        if not self.main_window or not hasattr(self.main_window, 'document_data'):
            return
        
        page_structures = self.main_window.document_data.get('page_structures', {})
        page_structure = page_structures.get(str(page_num), {})
        page_elements = page_structure.get('structure', {}).get('elements', [])
        logger.info(f"page_elements: {page_elements}")
        for element in page_elements:
            logger.info(f"{type(element)} {element.get('id')} {type(element.get('id'))}")
            if int(element.get('id')) == int(element_id):
                return element

    def update_content(self, page_structures):
        """Update the structured content view with page structures"""
        logger.debug("StructuredContentView.update_content called")
        logger.debug(f"Received page_structures with {len(page_structures)} pages")
        
        self._current_content = page_structures  # Store for view switching
        
        # Clear current view
        self.content_list_widget.clear()
        self.content_list = []
        
        # Group content by type
        #content_type_list = ['image', 'table', 'figure', 'picture']

        
        # Process each page's structure
        figure_number = 0
        for page_num, structure in page_structures.items():
            elements = structure.get('structure', {}).get('elements', [])
            logger.debug(f"Processing page {page_num} with {len(elements)} elements")
            for element in elements:
                element_type = element.get('category', '').lower()
                if element_type.startswith('figure:'):
                    # Find nearest caption
                    linked_elements = element.get('linked_elements', None)
                    caption_element = None
                    if linked_elements:
                        for linked_element in linked_elements:
                            element_info = self._get_element_info( linked_element[0], linked_element[1])
                            if element_info.get('category', '').lower() == 'caption':
                                caption_element = element_info
                                break
                    caption_text = caption_element.get('content', {}).get('text', '') if caption_element else None
                    item = {
                        'figure_element': element,
                        'caption_element': caption_element,
                        'figure_number': figure_number,
                        'category': element_type,
                        'caption': caption_text,
                        'coordinates': element.get('coordinates', []),
                        'id': element.get('id', ''),
                        'page_number': int(page_num),
                        'page_width': element.get('attributes', {}).get('page_width', 0),
                        'page_height': element.get('attributes', {}).get('page_height', 0)
                    }

                    self.content_list.append(item)
                    figure_number += 1
                    logger.debug(f"Added element of type {element_type} from page {page_num}")
        
        # Update view based on current mode
        if self.content_list_widget.viewMode() == QListWidget.ViewMode.ListMode:
            logger.debug("Updating list view")
            self._update_list_view()
        else:
            logger.debug("Updating icon view")
            self._update_icon_view()
    
    def _update_list_view(self):
        """Update the list view with content (compatible with new self.content_list format)"""
        self.content_list_widget.clear()
        for idx, item in enumerate(self.content_list):
            # Create item widget
            item_widget = QWidget()
            item_layout = QHBoxLayout()
            item_layout.setContentsMargins(5, 5, 5, 5)
            item_layout.setSpacing(10)

            # Add small thumbnail (if available)
            item_pixmap = self._get_cached_pixmap(item)
            if item_pixmap:
                thumb_label = QLabel()
                thumb_label.setPixmap(item_pixmap.scaled(48, 48, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                thumb_label.setFixedSize(52, 52)
                item_layout.addWidget(thumb_label)
            else:
                spacer = QLabel()
                spacer.setFixedSize(52, 52)
                item_layout.addWidget(spacer)

            # Add figure number and page number
            info_layout = QVBoxLayout()
            info_layout.setSpacing(2)
            fig_label = QLabel(f"Figure {idx+1}")
            fig_label.setStyleSheet("font-weight: bold;")
            info_layout.addWidget(fig_label)
            page_label = QLabel(f"Page {item['page_number']+1}")
            page_label.setStyleSheet("color: #555;")
            info_layout.addWidget(page_label)
            item_layout.addLayout(info_layout)

            # Add caption if available
            if item.get('caption'):
                caption_label = QLabel(item['caption'])
                caption_label.setWordWrap(True)
                caption_label.setStyleSheet("font-style: italic;")
                caption_label.setFixedWidth(300)
                item_layout.addWidget(caption_label)

            item_widget.setLayout(item_layout)
            item_widget.setStyleSheet("border-bottom: 1px solid #eee; padding: 2px;")

            # Store element data for double-click
            item_widget.element_data = item

            # Create list item
            list_item = QListWidgetItem()
            list_item.setSizeHint(item_widget.sizeHint())
            self.content_list_widget.addItem(list_item)
            self.content_list_widget.setItemWidget(list_item, item_widget)

    def _update_icon_view(self):
        """Update the icon view with content"""
        self.content_list_widget.clear()
        
        if not self.current_doc:
            logger.warning("No PDF document available for icon view")
            return
            
        # Add items in chronological order
        for idx, item in enumerate(self.content_list):
            # Create item widget
            item_widget = QWidget()
            item_layout = QVBoxLayout()
            item_layout.setContentsMargins(5, 5, 5, 5)
            
            # Get cached pixmap
            item_pixmap = self._get_cached_pixmap(item)
            
            if item_pixmap:
                # Create image label
                image_label = QLabel()
                image_label.setPixmap(item_pixmap)
                image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                item_layout.addWidget(image_label)
            
            # Add page number and type label
            header_label = QLabel(f"Figure {idx+1}.")
            header_label.setStyleSheet("font-weight: bold;")
            item_layout.addWidget(header_label)
            
            # Show caption if available
            if item['caption']:
                caption_label = QLabel(item['caption'])
                caption_label.setWordWrap(True)
                caption_label.setStyleSheet("font-style: italic;")
                item_layout.addWidget(caption_label)
            
            item_widget.setLayout(item_layout)
            item_widget.setStyleSheet("border: 1px solid #ccc; padding: 5px;")
            
            # Store element data for double-click
            item_widget.element_data = item
            
            # Create list item
            list_item = QListWidgetItem()
            list_item.setSizeHint(item_widget.sizeHint())
            self.content_list_widget.addItem(list_item)
            self.content_list_widget.setItemWidget(list_item, item_widget)
            
            # Add spacing between items
            spacer = QListWidgetItem()
            spacer.setSizeHint(QSize(0, 10))
            spacer.setFlags(Qt.ItemFlag.NoItemFlags)
            self.content_list_widget.addItem(spacer)
            
            # Process events to keep UI responsive
            QApplication.processEvents()

    def _clear_grid(self):
        """Clear the grid layout"""
        while self.content_grid.count():
            item = self.content_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def _is_caption_for_element(self, caption_element, target_element):
        """Check if a caption element belongs to a target element"""
        # Get coordinates
        caption_coords = caption_element.get('coordinates', [])
        target_coords = target_element.get('coordinates', [])
        
        if not caption_coords or not target_coords:
            return False
            
        # Get y-coordinates (vertical position)
        caption_y = caption_coords[0]['y']
        target_y = target_coords[0]['y']
        
        # Check if caption is below the target element
        # and within a reasonable distance (e.g., 0.1 of page height)
        return caption_y > target_y and (caption_y - target_y) < 0.1

    def _show_element_info(self, item):
        """Show element information dialog"""
        # Get the main window instance
        if not self.main_window:
            return
            
        # Get the widget associated with the item
        widget = self.content_list_widget.itemWidget(item)
        if not widget or not hasattr(widget, 'element_data'):
            return
            
        # Get the page number from the element data
        page_num = widget.element_data.get('page')
        logger.debug(f"Page number: {page_num}")
        if page_num is not None:
            # Update current page in PDF viewer
            self.main_window.current_page = page_num - 1  # Convert to 0-based index
            self.main_window.pdf_viewer.current_page = page_num - 1
            #main_window.pdf_viewer.display_all_pages()
            
            # Calculate the scroll position for the target page
            scroll_bar = self.main_window.pdf_scroll.verticalScrollBar()
            target_pos = 0
            
            # Sum up the heights of all pages before the target page
            for i in range(page_num - 1):
                if i in self.main_window.pdf_viewer.page_pixmaps:
                    target_pos += self.main_window.pdf_viewer.page_pixmaps[i]['height']
            
            # Scroll to the target position
            scroll_bar.setValue(target_pos)
            self.main_window.update_navigation()
            
        # Show element info dialog
        dialog = ElementInfoDialog(widget.element_data, self.main_window)
        dialog.exec()

    def scroll_to_page_element(self, page_num):
        """Scroll to the first element for the given 0-based page number (page_num)"""
        # Elements in content_list are added in _update_icon_view/_update_list_view
        # We need to find the first item whose element_data['page'] == page_num+1
        for i in range(self.content_list_widget.count()):
            item = self.content_list_widget.item(i)
            widget = self.content_list_widget.itemWidget(item)
            if widget and hasattr(widget, 'element_data'):
                element_page = widget.element_data.get('page')
                if element_page == page_num + 1:
                    self.content_list_widget.scrollToItem(item, QListWidget.ScrollHint.PositionAtCenter)
                    self.content_list_widget.setCurrentItem(item)
                    break

    def show_figures_from_db(self, document):
        """Display all figures from PrFigure for the given document."""
        self.content_list_widget.clear()
        figures = PrFigure.select().where(PrFigure.document == document).order_by(PrFigure.part1_prefix, PrFigure.part1_number, PrFigure.part2_prefix, PrFigure.part2_number)
        for fig in figures:
            item_widget = QWidget()
            layout = QVBoxLayout()
            layout.setContentsMargins(5, 5, 5, 5)

            # Figure image
            if fig.figure_binary:
                pixmap = QPixmap()
                pixmap.loadFromData(fig.figure_binary)
                image_label = QLabel()
                image_label.setPixmap(pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                layout.addWidget(image_label)

            # Figure number and page
            info_label = QLabel(f"{fig.part1_prefix}{fig.part1_number}{fig.part2_prefix or ''}{fig.part2_number or ''} (Page {fig.figure_page_number})")
            info_label.setStyleSheet("font-weight: bold;")
            layout.addWidget(info_label)

            # Caption text
            if fig.caption_text:
                caption_label = QLabel(fig.caption_text)
                caption_label.setWordWrap(True)
                caption_label.setStyleSheet("font-style: italic;")
                layout.addWidget(caption_label)

            # Caption image (optional)
            if fig.caption_binary:
                cap_pixmap = QPixmap()
                cap_pixmap.loadFromData(fig.caption_binary)
                cap_image_label = QLabel()
                cap_image_label.setPixmap(cap_pixmap.scaled(150, 50, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                layout.addWidget(cap_image_label)

            item_widget.setLayout(layout)
            list_item = QListWidgetItem()
            list_item.setSizeHint(item_widget.sizeHint())
            self.content_list_widget.addItem(list_item)
            self.content_list_widget.setItemWidget(list_item, item_widget)

class FigureGalleryWidget(QWidget):
    def __init__(self, main_window=None):
        super().__init__(None)
        self.figures = []  # List of dicts: {'pixmap': ..., 'bounding_boxes': ..., 'caption': ...}
        self._scroll_callback = None
        self.setMinimumWidth(10)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.selected_index = None  # Track selected figure index
        self.setMouseTracking(True)  # Enable mouse tracking for hover events
        self._selected_subfig_box = None  # Track selected subfigure box
        self.main_window = main_window
        self.show_subfigures = False  # Always hide subfigure boxes in gallery view
        self.selection_changed_callback = None  # Callback for selection sync
        self.subfigures_changed_callback = None  # Callback for subfigures detection
        self.add_subfigure_mode = False  # Track if in add subfigure mode
        self._add_subfig_start = None
        self._add_subfig_end = None
        self.subfig_numbering_scheme = '1,2,3'  # Default numbering scheme
        self._dragging_box = None
        self._pending_drag_box = None  # For robust drag detection
        self._pending_drag_offset = None
        self._drag_start_pos = None
        self._resizing_box = None  # Ensure always initialized

    def set_selected_index(self, idx, subfig_idx=None):
        #logger.info(f"[Gallery] set_selected_index: idx={idx}, subfig_idx={subfig_idx}, selected_index={self.selected_index}")
        self._selected_subfig_box = subfig_idx
        self.selected_index = idx

        if self.selection_changed_callback:
            self.selection_changed_callback(idx, subfig_idx)

    def update_height(self):
        width = self.width() if self.width() > 0 else 10
        total_height = 0
        for fig in self.figures:
            if fig.get('parent_id'):
                continue  # Skip subfigures
            pixmap = fig['pixmap']
            if pixmap.isNull():
                continue
            scaled_height = pixmap.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation).height()
            total_height += scaled_height + 40
        self.setFixedHeight(total_height)

    def set_figures(self, figures):
        '''
        figures is a list of PrFigure objects
        gallery_figures.append({'id': fig.id, 'pixmap': pixmap, 'caption': fig.caption_text or "", 
                        'parent_id': fig.parent.id if fig.parent else None, 'subfigure_box': fig.subfigure_box})
        '''
        # Build mapping from id to parent figure
        parents = {}
        for fig in figures:
            if fig.get('parent_id') is None:
                fig['subfig_boxes'] = []  # Initialize
                parents[fig['id']] = fig
        # Now, assign subfigures to their parent
        for fig in figures:
            if fig.get('parent_id') is not None:
                parent_id = fig['parent_id']
                parent = parents.get(parent_id)
                if parent is not None:
                    # Convert subfigure_box to QRect
                    try:
                        box = fig.get('subfigure_box')
                        if box:
                            if isinstance(box, str):
                                box = json.loads(box)
                            if isinstance(box, dict):
                                pixmap = parent.get('pixmap')
                                if pixmap:
                                    x = int(box['x'] * pixmap.width())
                                    y = int(box['y'] * pixmap.height())
                                    w = int(box['w'] * pixmap.width())
                                    h = int(box['h'] * pixmap.height())
                                    rect = QRect(x, y, w, h)
                                    subfigure = { 'id': fig.get('id'), 'box': rect }
                                    parent['subfig_boxes'].append(subfigure)
                    except Exception as e:
                        logger.error(f"Error converting subfigure_box to QRect for subfigure id={fig.get('id')}: {e}")
        # Only keep parent figures in self.figures
        self.figures = list(parents.values())
        self.selected_index = None
        self._selected_subfig_box = None
        self.update_height()
        self.update()

    def scroll_to_figure(self, index):
        # Calculate the y position of the top of the figure at the given index
        y = 0
        width = self.width() if self.width() > 0 else 400
        for i, fig in enumerate(self.figures):
            pixmap = fig['pixmap']
            if pixmap.isNull():
                continue
            scaled_height = pixmap.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation).height()
            if i == index:
                break
            y += scaled_height + 40
        # Use the callback to scroll the parent scroll area
        if self._scroll_callback:
            self._scroll_callback(y)

    def set_scroll_callback(self, callback):
        self._scroll_callback = callback

    def paintEvent(self, event):
        painter = QPainter(self)
        y = 0
        width = self.width()
        for i, fig in enumerate(self.figures):
            pixmap = fig['pixmap']
            if pixmap.isNull():
                continue
            if fig['parent_id']:
                continue
            scaled_pixmap = pixmap.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)
            painter.drawPixmap(0, y, scaled_pixmap)
            if i == self.selected_index:
                pen = QPen(QColor(255, 0, 0), 4)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(0, y, scaled_pixmap.width(), scaled_pixmap.height())
            show_subfigs = (i == self.selected_index) or self.show_subfigures
            if 'subfig_boxes' in fig:
                for idx, subfig in enumerate(fig['subfig_boxes']):
                    rect = subfig['box']
                    if show_subfigs:
                        pen = QPen(QColor(0, 200, 0), 3)
                        painter.setPen(pen)
                    else:
                        pen = QPen(QColor(0, 200, 0, 120),2)
                        painter.setPen(pen)
                    x_scale = scaled_pixmap.width() / pixmap.width()
                    y_scale = scaled_pixmap.height() / pixmap.height()
                    sx = int(rect.x() * x_scale)
                    sy = int(rect.y() * y_scale)
                    sw = int(rect.width() * x_scale)
                    sh = int(rect.height() * y_scale)
                    if i == self.selected_index and hasattr(self, '_selected_subfig_box') and idx == self._selected_subfig_box:
                        painter.setBrush(QColor(0, 200, 0, 80))
                    else:
                        painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawRect(sx, y + sy, sw, sh)
                    center_x = sx + sw // 2
                    center_y = y + sy + sh // 2
                    font = painter.font()
                    font.setBold(True)
                    font.setPointSize(12)
                    painter.setFont(font)
                    painter.setPen(QColor(0, 0, 0))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    text = self._get_subfig_label(idx)
                    text_rect = painter.boundingRect(QRect(center_x - 20, center_y - 12, 40, 24), Qt.AlignmentFlag.AlignCenter, text)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor(255, 255, 255, 200))
                    painter.drawRect(text_rect)
                    painter.setPen(QColor(0, 0, 0))
                    painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, text)
            if i == self.selected_index and self.add_subfigure_mode and self._add_subfig_start and self._add_subfig_end:
                pen = QPen(QColor(0, 0, 255), 2, Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                x1, y1 = self._add_subfig_start.x(), self._add_subfig_start.y()
                x2, y2 = self._add_subfig_end.x(), self._add_subfig_end.y()
                rect = QRect(int(min(x1, x2)), int(min(y1, y2)), int(abs(x2 - x1)), int(abs(y2 - y1)))
                painter.drawRect(rect)
            if fig.get('caption'):
                painter.setPen(Qt.GlobalColor.black)
                painter.drawText(0, y + scaled_pixmap.height() + 20, fig['caption'])
            y += scaled_pixmap.height() + 40

    def sizeHint(self):
        width = self.width() if self.width() > 0 else 10
        total_height = 0
        for fig in self.figures:
            if fig.get('parent_id'):
                continue  # Skip subfigures
            pixmap = fig['pixmap']
            if pixmap.isNull():
                continue
            scaled_height = pixmap.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation).height()
            total_height += scaled_height + 40
        return QSize(width, total_height)

    def minimumSizeHint(self):
        width = self.width() if self.width() > 0 else 10
        return QSize(width, 0)

    def resizeEvent(self, event):
        self.update_height()
        super().resizeEvent(event)

    def contextMenuEvent(self, event):
        y = event.position().y() if hasattr(event, 'position') else event.y()
        width = self.width()
        current_y = 0
        clicked_index = None
        for i, fig in enumerate(self.figures):
            pixmap = fig['pixmap']
            if pixmap.isNull():
                continue
            scaled_pixmap = pixmap.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)
            if current_y <= y < current_y + scaled_pixmap.height():
                clicked_index = i
                break
            current_y += scaled_pixmap.height() + 40
        if clicked_index is not None and clicked_index == self.selected_index:
            menu = QMenu(self)
            edit_action = menu.addAction("Edit")
            detect_action = menu.addAction("Detect subfigures")
            detect_action.triggered.connect(lambda: self.detect_subfigures(clicked_index))
            menu.exec(event.globalPos())

    def detect_subfigures(self, idx):
        pixmap = self.figures[idx]['pixmap']
        result, annotated_pixmap = self.segment_figures_qt(pixmap)
        # Store bounding boxes in the figure's data
        boxes = [rect for _, rect in result]
        for box in boxes:
            subfigure = { 'id': None, 'box': box }
            self.figures[idx]['subfig_boxes'].append(subfigure)
        self.update()
        # Notify parent that subfigures have changed
        if self.subfigures_changed_callback:
            self.subfigures_changed_callback()

    def check_overlap(self, box1, box2):
        # box = (x, y, w, h)
        x1, y1, w1, h1 = box1[:4]
        x2, y2, w2, h2 = box2[:4]
        if (x1 < x2 + w2 and x1 + w1 > x2 and
            y1 < y2 + h2 and y1 + h1 > y2):
            return True
        return False

    def segment_figures_qt(self, qpixmap):
        img = qpixmap.toImage()
        width = img.width()
        height = img.height()
        ptr = img.constBits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        img = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        original_img = img.copy()
        scale_factor = max(1, min(width, height) / 4000)
        proc_width = int(width / scale_factor)
        proc_height = int(height / scale_factor)
        proc_img = cv2.resize(img, (proc_width, proc_height))
        gray = cv2.cvtColor(proc_img, cv2.COLOR_RGB2GRAY)
        _, black_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        proc_img[black_mask == 0] = [255, 255, 255]
        def process_image(img):
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            kernel = np.ones((5,5), np.uint8)
            gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            total_area = img.shape[0] * img.shape[1]
            min_contour_area = total_area * 0.005
            max_contour_area = total_area * 0.8
            bounding_boxes = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_contour_area < area < max_contour_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    bounding_boxes.append((x, y, w, h))
            return bounding_boxes
        bounding_boxes = process_image(proc_img)
        if len(bounding_boxes) < 10 or len(bounding_boxes) > 50:
            low, high = 0.001, 0.05
            while high - low > 0.0001:
                mid = (low + high) / 2
                min_contour_area = proc_width * proc_height * mid
                bounding_boxes = process_image(proc_img)
                if len(bounding_boxes) < 10:
                    high = mid
                elif len(bounding_boxes) > 50:
                    low = mid
                else:
                    break
        bounding_boxes = [(int(x*scale_factor), int(y*scale_factor), 
                        int(w*scale_factor), int(h*scale_factor)) 
                        for x, y, w, h in bounding_boxes]
        valid_boxes = []
        for box in bounding_boxes:
            is_valid = True
            for valid_box in valid_boxes:
                if self.check_overlap(box, valid_box):
                    if box[2] * box[3] > valid_box[2] * valid_box[3]:
                        valid_boxes.remove(valid_box)
                    else:
                        is_valid = False
                    break
            if is_valid:
                valid_boxes.append(box)
        logger.info(f"Detected {len(valid_boxes)} subfigures (bounding boxes)")
        for i, box in enumerate(valid_boxes):
            logger.info(f"Box {i+1}: x={box[0]}, y={box[1]}, w={box[2]}, h={box[3]}")
        if valid_boxes:
            avg_height = sum(box[3] for box in valid_boxes) / len(valid_boxes)
            for i, box in enumerate(valid_boxes):
                box_y = box[1]
                row = int(box_y / (avg_height * 1.2))
                valid_boxes[i] = box + (row,)
            valid_boxes.sort(key=lambda box: (box[4], box[0]))
        annotated_pixmap = qpixmap.copy()
        painter = QPainter(annotated_pixmap)
        painter.setPen(QPen(Qt.GlobalColor.green, 5, Qt.PenStyle.SolidLine))
        result = []
        for i, (x, y, w, h, _) in enumerate(valid_boxes, start=1):
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(original_img.shape[1] - x, w + 2*padding)
            h = min(original_img.shape[0] - y, h + 2*padding)
            figure = original_img[y:y+h, x:x+w]
            height_, width_, channel = figure.shape
            bytes_per_line = 3 * width_
            q_img = QImage(figure.tobytes(), width_, height_, bytes_per_line, QImage.Format.Format_RGB888)
            cropped_pixmap = QPixmap.fromImage(q_img)
            result.append((cropped_pixmap, QRect(x, y, w, h)))
            painter.drawRect(x, y, w, h)
        painter.end()
        return result, annotated_pixmap

    # --- Basic box editing: click and drag to move a box ---
    def mousePressEvent(self, event):
        #logger.info(f"[Gallery] mousePressEvent: pos=({event.position().x() if hasattr(event, 'position') else event.x()}, {event.position().y() if hasattr(event, 'position') else event.y()})")
        # Add subfigure mode: start drawing bounding box
        if self.add_subfigure_mode and self.selected_index is not None:
            #logger.info(f"[Gallery] add_subfigure_mode active, selected_index={self.selected_index}")
            self._add_subfig_start = event.position() if hasattr(event, 'position') else QPointF(event.x(), event.y())
            self._add_subfig_end = self._add_subfig_start
            self.update()
            return  # Do not process any other logic when adding a subfigure
        # Allow selecting a figure by clicking anywhere on its image (not just subfigures)
        y = event.position().y() if hasattr(event, 'position') else event.y()
        width = self.width()
        current_y = 0
        clicked_index = None
        for i, fig in enumerate(self.figures):
            if fig.get('parent_id'):
                continue  # Skip subfigures
            pixmap = fig['pixmap']
            if pixmap.isNull():
                continue
            scaled_pixmap = pixmap.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)
            if current_y <= y < current_y + scaled_pixmap.height():
                clicked_index = i
                break
            current_y += scaled_pixmap.height() + 40
        logger.info(f"[Gallery] clicked_index={clicked_index}")
        # Calculate y_offset for the selected main figure only (skip subfigures)
        y_offset = 0
        for i, fig in enumerate(self.figures):
            if fig.get('parent_id'):
                continue
            if i == clicked_index:
                break
            p = fig['pixmap']
            y_offset += p.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation).height() + 40
        # --- PRIORITIZE RESIZING ---
        if clicked_index is not None:
            fig = self.figures[clicked_index]
            if 'subfig_boxes' in fig:
                pixmap = fig['pixmap']
                scaled_pixmap = pixmap.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)
                x_scale = scaled_pixmap.width() / pixmap.width()
                y_scale = scaled_pixmap.height() / pixmap.height()
                click_x = event.position().x() if hasattr(event, 'position') else event.x()
                click_y = (event.position().y() if hasattr(event, 'position') else event.y()) - y_offset
                edge_threshold = 8
                # --- Check for resize zone first (priority) ---
                for idx, subfig in enumerate(fig['subfig_boxes']):
                    rect = subfig['box']
                    sx = int(rect.x() * x_scale)
                    sy = int(rect.y() * y_scale)
                    sw = int(rect.width() * x_scale)
                    sh = int(rect.height() * y_scale)
                    near_left = abs(click_x - sx) <= edge_threshold
                    near_right = abs(click_x - (sx + sw)) <= edge_threshold
                    near_top = abs(click_y - sy) <= edge_threshold
                    near_bottom = abs(click_y - (sy + sh)) <= edge_threshold
                    inside_x = sx < click_x < sx + sw
                    inside_y = sy < click_y < sy + sh
                    resize_edges = None
                    if (near_left and near_top):
                        resize_edges = 'nw'
                    elif (near_right and near_top):
                        resize_edges = 'ne'
                    elif (near_left and near_bottom):
                        resize_edges = 'sw'
                    elif (near_right and near_bottom):
                        resize_edges = 'se'
                    elif near_left and inside_y:
                        resize_edges = 'w'
                    elif near_right and inside_y:
                        resize_edges = 'e'
                    elif near_top and inside_x:
                        resize_edges = 'n'
                    elif near_bottom and inside_x:
                        resize_edges = 's'
                    # If in a resize zone, immediately enter resizing mode (priority)
                    if resize_edges:
                        logger.info(f"[Gallery] PRIORITY: Clicked resize zone for subfigure idx={idx} in figure {clicked_index}, resize_edges={resize_edges}")
                        self.set_selected_index(clicked_index, idx)
                        self._resizing_box = idx
                        self._resize_edges = resize_edges
                        self._resize_start = (click_x, click_y)
                        self._resize_rect_orig = subfig['box']
                        self.update()
                        return  # Do not fall through to selection logic
                # --- If not in a resize zone, proceed to normal selection logic ---
                for idx, subfig in enumerate(fig['subfig_boxes']):
                    rect = subfig['box']
                    sx = int(rect.x() * x_scale)
                    sy = int(rect.y() * y_scale)
                    sw = int(rect.width() * x_scale)
                    sh = int(rect.height() * y_scale)
                    if sx <= click_x <= sx + sw and sy <= click_y <= sy + sh:
                        #logger.info(f"[Gallery] Clicked subfigure idx={idx} in figure {clicked_index}, resize_edges=None")
                        self.set_selected_index(clicked_index, idx)
                        self._pending_drag_box = idx
                        self._pending_drag_offset = (click_x - sx, click_y - sy)
                        self._drag_start_pos = (click_x, click_y)
                        self.update()
                        return  # Do not clear selection/drag state after this
        # If not clicking a subfigure, select main figure and clear subfig selection/drag state
        if clicked_index is not None:
            #logger.info(f"[Gallery] set_selected_index({clicked_index}) called (main figure click)")
            self.set_selected_index(clicked_index, None)
            self._selected_subfig_box = None
            self._pending_drag_box = None
            self._dragging_box = None
            self._resizing_box = None
            self._drag_start_pos = None
            self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.add_subfigure_mode and self._add_subfig_start is not None:
            self._add_subfig_end = event.position() if hasattr(event, 'position') else QPointF(event.x(), event.y())
            self.update()
            return
        width = self.width()
        cursor_set = False
        if self.selected_index is not None:
            fig = self.figures[self.selected_index] 
            if 'subfig_boxes' in fig and hasattr(self, '_selected_subfig_box') and self._selected_subfig_box is not None:
                y_offset = 0
                for i, f in enumerate(self.figures):
                    if f.get('parent_id'):
                        continue
                    if i == self.selected_index:
                        break
                    p = f['pixmap']
                    y_offset += p.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation).height() + 40
                pixmap = fig['pixmap']
                scaled_pixmap = pixmap.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)
                fig_height = scaled_pixmap.height()
                mouse_y = event.position().y() if hasattr(event, 'position') else event.y()
                if y_offset <= mouse_y < y_offset + fig_height:
                    move_x = event.position().x() if hasattr(event, 'position') else event.x()
                    move_y = mouse_y - y_offset
                    x_scale = scaled_pixmap.width() / pixmap.width()
                    y_scale = scaled_pixmap.height() / pixmap.height()
                    edge_threshold = 8
                    subfig = fig['subfig_boxes'][self._selected_subfig_box]
                    rect = subfig['box']
                    sx = int(rect.x() * x_scale)
                    sy = int(rect.y() * y_scale)
                    sw = int(rect.width() * x_scale)
                    sh = int(rect.height() * y_scale)
                    near_left = abs(move_x - sx) <= edge_threshold
                    near_right = abs(move_x - (sx + sw)) <= edge_threshold
                    near_top = abs(move_y - sy) <= edge_threshold
                    near_bottom = abs(move_y - (sy + sh)) <= edge_threshold
                    inside_x = sx < move_x < sx + sw
                    inside_y = sy < move_y < sy + sh
                    if (near_left and near_top) or (near_right and near_bottom):
                        self.setCursor(Qt.CursorShape.SizeFDiagCursor)
                        cursor_set = True
                    elif (near_right and near_top) or (near_left and near_bottom):
                        self.setCursor(Qt.CursorShape.SizeBDiagCursor)
                        cursor_set = True
                    elif (near_left and inside_y) or (near_right and inside_y):
                        self.setCursor(Qt.CursorShape.SizeHorCursor)
                        cursor_set = True
                    elif (near_top and inside_x) or (near_bottom and inside_x):
                        self.setCursor(Qt.CursorShape.SizeVerCursor)
                        cursor_set = True
        if not cursor_set:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        if self._pending_drag_box is not None and event.buttons() & Qt.MouseButton.LeftButton:
            move_x = event.position().x() if hasattr(event, 'position') else event.x()
            move_y = event.position().y() if hasattr(event, 'position') else event.y()
            start_x, start_y = self._drag_start_pos if self._drag_start_pos else (move_x, move_y)
            drag_dist = ((move_x - start_x) ** 2 + (move_y - start_y) ** 2) ** 0.5
            if drag_dist > 4:
                self._dragging_box = self._pending_drag_box
                self._drag_offset = self._pending_drag_offset
                self._pending_drag_box = None
                self._pending_drag_offset = None
                self._drag_start_pos = None
        if hasattr(self, '_resizing_box') and self._resizing_box is not None and self.selected_index is not None:
            fig = self.figures[self.selected_index]
            if 'subfig_boxes' not in fig:
                return
            y_offset = 0
            for i, f in enumerate(self.figures):
                if f.get('parent_id'):
                    continue
                if i == self.selected_index:
                    break
                p = f['pixmap']
                y_offset += p.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation).height() + 40
            pixmap = fig['pixmap']
            scaled_pixmap = pixmap.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)
            x_scale = pixmap.width() / scaled_pixmap.width()
            y_scale = pixmap.height() / scaled_pixmap.height()
            move_x = event.position().x() if hasattr(event, 'position') else event.x()
            move_y = (event.position().y() if hasattr(event, 'position') else event.y()) - y_offset
            start_x, start_y = self._resize_start
            start_y = start_y - y_offset
            rect = self._resize_rect_orig
            new_rect = QRect(rect)
            dx = int((move_x - start_x) * x_scale)
            dy = int((move_y - start_y) * y_scale)
            move_x_pixmap = int(move_x * x_scale)
            move_y_pixmap = int(move_y * y_scale)
            if 'n' in self._resize_edges:
                new_rect.setTop(move_y_pixmap)
            if 's' in self._resize_edges:
                new_rect.setBottom(move_y_pixmap)
            if 'w' in self._resize_edges:
                new_rect.setLeft(move_x_pixmap)
            if 'e' in self._resize_edges:
                new_rect.setRight(move_x_pixmap)
            min_size = 10
            if new_rect.width() < min_size:
                if 'w' in self._resize_edges:
                    new_rect.setLeft(new_rect.right() - min_size)
                else:
                    new_rect.setRight(new_rect.left() + min_size)
            if new_rect.height() < min_size:
                if 'n' in self._resize_edges:
                    new_rect.setTop(new_rect.bottom() - min_size)
                else:
                    new_rect.setBottom(new_rect.top() + min_size)
            fig['subfig_boxes'][self._resizing_box]['box'] = new_rect
            self.update()
        elif self._dragging_box is not None and self.selected_index is not None:
            fig = self.figures[self.selected_index]
            if 'subfig_boxes' not in fig:
                return
            y_offset = 0
            for i, f in enumerate(self.figures):
                if f.get('parent_id'):
                    continue
                if i == self.selected_index:
                    break
                p = f['pixmap']
                y_offset += p.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation).height() + 40
            pixmap = fig['pixmap']
            scaled_pixmap = pixmap.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)
            x_scale = scaled_pixmap.width() / pixmap.width()
            y_scale = scaled_pixmap.height() / pixmap.height()
            move_x = event.position().x() if hasattr(event, 'position') else event.x()
            move_y = (event.position().y() if hasattr(event, 'position') else event.y()) - y_offset
            dx, dy = self._drag_offset
            rect = fig['subfig_boxes'][self._dragging_box]['box']
            new_x = int((move_x - dx) / x_scale)
            new_y = int((move_y - dy) / y_scale)
            fig['subfig_boxes'][self._dragging_box]['box'] = QRect(new_x, new_y, rect.width(), rect.height())
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.add_subfigure_mode and self.selected_index is not None and self._add_subfig_start is not None:
            # Finish drawing bounding box
            start = self._add_subfig_start
            end = self._add_subfig_end if self._add_subfig_end is not None else start
            width = self.width()
            # Calculate y_offset for the selected main figure only (skip subfigures)
            y_offset = 0
            for i, fig in enumerate(self.figures):
                if fig.get('parent_id'):
                    continue
                if i == self.selected_index:
                    break
                p = fig['pixmap']
                y_offset += p.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation).height() + 40
            fig = self.figures[self.selected_index]
            pixmap = fig['pixmap']
            scaled_pixmap = pixmap.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)
            # Get coordinates relative to the top-left of the selected figure
            x1, y1 = start.x(), start.y() - y_offset
            x2, y2 = end.x(), end.y() - y_offset
            x_min, y_min = min(x1, x2), min(y1, y2)
            x_max, y_max = max(x1, x2), max(y1, y2)
            # Clamp to figure area
            x_min = max(0, min(x_min, scaled_pixmap.width()))
            x_max = max(0, min(x_max, scaled_pixmap.width()))
            y_min = max(0, min(y_min, scaled_pixmap.height()))
            y_max = max(0, min(y_max, scaled_pixmap.height()))
            # Convert to original pixmap coordinates
            x_scale = pixmap.width() / scaled_pixmap.width()
            y_scale = pixmap.height() / scaled_pixmap.height()
            px = int(x_min * x_scale)
            py = int(y_min * y_scale)
            pw = int((x_max - x_min) * x_scale)
            ph = int((y_max - y_min) * y_scale)
            if pw > 10 and ph > 10:
                rect = QRect(px, py, pw, ph)
                if 'subfig_boxes' not in fig:
                    fig['subfig_boxes'] = []
                fig['subfig_boxes'].append({'id': None, 'box': rect})
                self._selected_subfig_box = len(fig['subfig_boxes']) - 1
                if self.subfigures_changed_callback:
                    self.subfigures_changed_callback()
            self._add_subfig_start = None
            self._add_subfig_end = None
            self.add_subfigure_mode = False
            self.update()
            return
        # Only select a subfigure if not dragging or resizing
        dragging = self._dragging_box is not None
        resizing = self._resizing_box is not None
        if self.selected_index is not None and not dragging and not resizing:
            width = self.width()
            y_offset = 0
            for i, fig in enumerate(self.figures):
                if fig.get('parent_id'):
                    continue
                if i == self.selected_index:
                    break
                p = fig['pixmap']
                y_offset += p.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation).height() + 40
            fig = self.figures[self.selected_index]
            if 'subfig_boxes' in fig:
                pixmap = fig['pixmap']
                scaled_pixmap = pixmap.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)
                x_scale = scaled_pixmap.width() / pixmap.width()
                y_scale = scaled_pixmap.height() / pixmap.height()
                release_x = event.position().x() if hasattr(event, 'position') else event.x()
                release_y = (event.position().y() if hasattr(event, 'position') else event.y()) - y_offset
                found = False
                for idx, subfig in enumerate(fig['subfig_boxes']):
                    rect = subfig['box']
                    sx = int(rect.x() * x_scale)
                    sy = int(rect.y() * y_scale)
                    sw = int(rect.width() * x_scale)
                    sh = int(rect.height() * y_scale)
                    if sx <= release_x <= sx + sw and sy <= release_y <= sy + sh:
                        self._selected_subfig_box = idx
                        found = True
                        self.update()
                        break
                if not found:
                    self._selected_subfig_box = None
                    self.update()
        if hasattr(self, '_dragging_box'):
            self._dragging_box = None
        if hasattr(self, '_resizing_box'):
            self._resizing_box = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)

    def save_subfigures(self):
        """Save subfigures of the selected figure to the PrFigure table, using the selected numbering scheme."""
        try:
            logger.info("Starting save_subfigures")
            if self.main_window and hasattr(self.main_window, 'document_record'):
                doc = self.main_window.document_record
                logger.info(f"Found document_record: {doc}")
            if not doc:
                logger.warning("No PDF document loaded.")
                QMessageBox.warning(self, "No Document", "No PDF document loaded.")
                return
            if self.selected_index is None:
                logger.warning("No figure selected.")
                QMessageBox.warning(self, "No Figure Selected", "Please select a figure to save its subfigures.")
                return
            fig = self.figures[self.selected_index]
            logger.info(f"Selected figure index: {self.selected_index}, fig: {fig}")
            if 'subfig_boxes' not in fig or not fig['subfig_boxes']:
                logger.warning("No subfigures detected for the selected figure.")
                QMessageBox.warning(self, "No Subfigures", "No subfigures detected for the selected figure.")
                return
            logger.info(f"Number of subfig_boxes: {len(fig['subfig_boxes'])}")
            # Remove previous subfigures for this parent
            logger.info("Querying for parent figure in PrFigure table...")
            parent_fig = PrFigure.select().where(PrFigure.id == fig['id']).first()
            if not parent_fig:
                # error and exit
                logger.error("Parent figure not found, exiting.")
                QMessageBox.critical(self, "Parent Figure Not Found", "Parent figure not found, exiting.")
                return
            else:
                logger.info(f"Found parent_fig: {parent_fig}")
            logger.info("Deleting old subfigures for this parent...")
            PrFigure.delete().where((PrFigure.document == doc) & (PrFigure.parent == parent_fig)).execute()
            logger.info("Saving each subfigure...")
            saved_count = 0
            for idx, subfig in enumerate(fig['subfig_boxes']):
                logger.info(f"Processing subfigure {idx+1}: subfig={subfig}")
                try:
                    rect = subfig['box']
                    pixmap = fig['pixmap']
                    x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
                    # Store bounding box as fractions of parent figure size
                    fig_w, fig_h = pixmap.width(), pixmap.height()
                    x_frac = round(x / fig_w, 6) if fig_w else 0.0
                    y_frac = round(y / fig_h, 6) if fig_h else 0.0
                    w_frac = round(w / fig_w, 6) if fig_w else 0.0
                    h_frac = round(h / fig_h, 6) if fig_h else 0.0
                    bounding_box_json = json.dumps({'x': x_frac, 'y': y_frac, 'w': w_frac, 'h': h_frac})
                    #logger.info(f"Bounding box JSON (fractional): {bounding_box_json}")
                    #logger.info(f"Cropping pixmap at ({x}, {y}, {w}, {h})")
                    cropped = pixmap.copy(int(x), int(y), int(w), int(h))
                    subfig_width, subfig_height = cropped.width(), cropped.height()
                    if cropped.isNull():
                        logger.error(f"Cropped pixmap is null for subfigure {idx+1}")
                        continue
                    logger.debug(f"Cropped pixmap size: {cropped.width()}x{cropped.height()}")
                    qimg = cropped.toImage()
                    if qimg.isNull():
                        logger.error(f"QImage is null for subfigure {idx+1}")
                        continue
                    bpp = qimg.depth() // 8
                    logger.debug(f"QImage depth: {qimg.depth()}, bytes per pixel: {bpp}")
                    expected_len = cropped.width() * cropped.height() * bpp
                    bits = qimg.bits()
                    bits.setsize(expected_len)
                    ba = bits.asstring(expected_len)
                    logger.debug(f"Buffer length: {len(ba)}, expected: {expected_len}")
                    buffer = io.BytesIO()
                    try:
                        pil_img = PILImage.frombytes("RGBA", (cropped.width(), cropped.height()), ba)
                    except Exception as pil_e:
                        logger.error(f"PIL.Image.frombytes failed for subfigure {idx+1}: {pil_e}")
                        continue
                    try:
                        pil_img.save(buffer, format="PNG")
                    except Exception as pil_save_e:
                        logger.error(f"PIL.Image.save failed for subfigure {idx+1}: {pil_save_e}")
                        continue
                    figure_binary = buffer.getvalue()
                    # Use numbering scheme for part2_number
                    part2_number = self._get_subfig_label(idx)
                    #logger.info(f"Saving PrFigure.create for subfigure {idx+1}, part2_number: {part2_number}, binary size: {len(figure_binary)}, bounding_box: {bounding_box_json}")
                    subfigure = PrFigure.create(
                        document=doc,
                        figure_number=str(self.selected_index+1),
                        figure_page_number=parent_fig.figure_page_number,
                        figure_element_id=parent_fig.figure_element_id,
                        part1_prefix=parent_fig.part1_prefix,
                        part1_number=parent_fig.part1_number,
                        part2_prefix=parent_fig.part2_prefix,
                        part2_number=part2_number,
                        figure_binary=figure_binary,
                        caption_binary=None,
                        caption_text=None,
                        parent=parent_fig,
                        created_at=datetime.datetime.now(),
                        updated_at=datetime.datetime.now(),
                        subfigure_box=bounding_box_json,
                        figure_width=subfig_width,
                        figure_height=subfig_height
                    )
                    subfigure.update_figure_number()
                    saved_count += 1
                except Exception as e:
                    logger.error(f"Error processing subfigure {idx+1}: {e}", exc_info=True)
                    continue
            self.main_window.figure_view.show_figures_from_db(doc)
            logger.info(f"Saved {saved_count} subfigures for {parent_fig.figure_number}.")
            QMessageBox.information(self, "Subfigures Saved", f"Saved {saved_count} subfigures for {parent_fig.figure_number}.")
            # Disable Save Subfigures button after saving
            if self.subfigures_changed_callback:
                self.subfigures_changed_callback()
        except Exception as e:
            logger.error(f"Error saving subfigures: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Save Error", f"Error saving subfigures: {str(e)}")

    def add_save_subfigures_button(self, layout):
        btn = QPushButton("Save Subfigures")
        btn.clicked.connect(self.save_subfigures)
        layout.addWidget(btn)

    def set_subfig_numbering_scheme(self, scheme):
        self.subfig_numbering_scheme = scheme
        self.update()

    def _get_subfig_label(self, idx):
        # idx is 0-based
        scheme = getattr(self, 'subfig_numbering_scheme', '1,2,3')
        if scheme == '1,2,3':
            return str(idx + 1)
        elif scheme == 'A,B,C':
            # A, B, ..., Z, AA, AB, ...
            n = idx
            label = ''
            while True:
                label = chr(ord('A') + (n % 26)) + label
                n = n // 26 - 1
                if n < 0:
                    break
            return label
        elif scheme == 'a,b,c':
            n = idx
            label = ''
            while True:
                label = chr(ord('a') + (n % 26)) + label
                n = n // 26 - 1
                if n < 0:
                    break
            return label
        else:
            return str(idx + 1)

    def mouseDoubleClickEvent(self, event):
        # Only allow if a figure and subfigure are selected and not dragging/resizing
        if self.selected_index is None:
            return
        # Prevent dialog if dragging or resizing is in progress
        if hasattr(self, '_dragging_box') and self._dragging_box is not None:
            return
        if hasattr(self, '_resizing_box') and self._resizing_box is not None:
            return
        fig = self.figures[self.selected_index]
        if 'subfig_boxes' not in fig:
            return
        width = self.width()
        pixmap = fig['pixmap']
        scaled_pixmap = pixmap.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)
        x_scale = scaled_pixmap.width() / pixmap.width()
        y_scale = scaled_pixmap.height() / pixmap.height()
        y_offset = 0
        for i in range(self.selected_index):
            p = self.figures[i]['pixmap']
            y_offset += p.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation).height() + 40
        click_x = event.position().x() if hasattr(event, 'position') else event.x()
        click_y = (event.position().y() if hasattr(event, 'position') else event.y()) - y_offset
        for idx, subfig in enumerate(fig['subfig_boxes']):
            rect = subfig['box']
            sx = int(rect.x() * x_scale)
            sy = int(rect.y() * y_scale)
            sw = int(rect.width() * x_scale)
            sh = int(rect.height() * y_scale)
            if sx <= click_x <= sx + sw and sy <= click_y <= sy + sh:
                current_label = self._get_subfig_label(idx)
                if hasattr(self, '_dragging_box'):
                    self._dragging_box = None
                if hasattr(self, '_resizing_box'):
                    self._resizing_box = None
                self.setCursor(Qt.CursorShape.ArrowCursor)
                new_label, ok = QInputDialog.getText(self, "Change Subfigure Index", f"Enter new index for this subfigure (current: {current_label}):", text=current_label)
                if ok and new_label and new_label != current_label:
                    new_idx = self._label_to_index(new_label)
                    if new_idx is not None and 0 <= new_idx < len(fig['subfig_boxes']):
                        box = fig['subfig_boxes'].pop(idx)
                        fig['subfig_boxes'].insert(new_idx, box)
                        self._selected_subfig_box = new_idx
                        self.update()
                    else:
                        QMessageBox.warning(self, "Invalid Index", f"Index '{new_label}' is not valid for the current numbering scheme.")
                break
        super().mouseDoubleClickEvent(event)

    def _label_to_index(self, label):
        # Convert a label (e.g., 'C', 'b', '12') to a 0-based index according to the current scheme
        scheme = getattr(self, 'subfig_numbering_scheme', '1,2,3')
        if scheme == '1,2,3':
            try:
                idx = int(label) - 1
                return idx if idx >= 0 else None
            except Exception:
                return None
        elif scheme == 'A,B,C':
            # Support A, B, ..., Z, AA, AB, ...
            label = label.strip().upper()
            idx = 0
            for c in label:
                if not ('A' <= c <= 'Z'):
                    return None
                idx = idx * 26 + (ord(c) - ord('A') + 1)
            return idx - 1 if idx > 0 else None
        elif scheme == 'a,b,c':
            label = label.strip().lower()
            idx = 0
            for c in label:
                if not ('a' <= c <= 'z'):
                    return None
                idx = idx * 26 + (ord(c) - ord('a') + 1)
            return idx - 1 if idx > 0 else None
        else:
            return None

    def _on_detect_subfigures_clicked(self):
        # Detect subfigures for the currently selected figure
        idx = self.selected_index
        if idx is not None:
            self.detect_subfigures(idx)

class PrFigureTableModel(QAbstractTableModel):
    def __init__(self, figures=None, parent=None):
        super().__init__(parent)
        self._figures = figures or []
        self._headers = ['Image', 'Figure #', 'Page', "Prefix1", "#1", "Sep.", "Prefix2", "#2", "Subfigs.", 'Caption']

    def rowCount(self, parent=QModelIndex()):
        return len(self._figures)

    def columnCount(self, parent=QModelIndex()):
        return len(self._headers)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return QVariant()
        fig = self._figures[index.row()]
        col = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            if col == 1:
                return str(fig.figure_number)
            elif col == 2:
                return str(fig.figure_page_number)
            elif col == 3:
                return fig.part1_prefix or ''
            elif col == 4:
                return fig.part1_number or ''
            elif col == 5:
                return fig.part_separator or ''
            elif col == 6:
                return fig.part2_prefix or ''
            elif col == 7:
                return fig.part2_number or ''
            elif col == 8:
                return len(fig.children)
            elif col == 9:
                return fig.caption_text or ''
        if role == Qt.ItemDataRole.DecorationRole and col == 0:
            if fig.figure_binary:
                pixmap = QPixmap()
                pixmap.loadFromData(fig.figure_binary)
                return pixmap
        return QVariant()

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._headers[section]
        return QVariant()

    def get_figure(self, row):
        if 0 <= row < len(self._figures):
            return self._figures[row]
        return None

    def set_figures(self, figures):
        self.beginResetModel()
        self._figures = figures
        self.endResetModel()

class FigureView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.model = PrFigureTableModel([])
        self.show_db_subfigures = True  # Track whether to show DB subfigures
        self.subfig_numbering_scheme = '1,2,3'  # Default numbering scheme
        self._syncing_selection = False  # Prevent recursive selection sync

        self.splitter = QSplitter(Qt.Orientation.Vertical, self)
        # Top: Gallery in scroll area
        self.gallery_widget = FigureGalleryWidget(self.main_window)
        self.gallery_widget.set_scroll_callback(self._scroll_gallery_to_y)
        self.gallery_widget.selection_changed_callback = self._on_gallery_selection_changed
        self.gallery_widget.subfigures_changed_callback = self._on_subfigures_changed  # New callback
        gallery_container = QWidget()
        gallery_layout = QVBoxLayout(gallery_container)
        gallery_layout.setContentsMargins(0, 0, 0, 0)
        gallery_layout.setSpacing(0)
        gallery_layout.addWidget(self.gallery_widget)
        self.gallery_scroll = QScrollArea()
        self.gallery_scroll.setWidgetResizable(True)
        self.gallery_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.gallery_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.gallery_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.gallery_scroll.setViewportMargins(0, 0, 0, 0)
        self.gallery_scroll.setWidget(gallery_container)
        # Add Add Subfigure, Save Subfigures, and Show/Hide Subfigures buttons below the gallery scroll area
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self.gallery_scroll)
        btn_row = QHBoxLayout()
        # --- Add numbering scheme combobox ---
        self.subfig_numbering_combo = QComboBox()
        self.subfig_numbering_combo.addItems(["1, 2, 3...", "A, B, C...", "a, b, c..."])
        self.subfig_numbering_combo.setCurrentIndex(0)
        self.subfig_numbering_combo.setToolTip("Subfigure numbering scheme")
        btn_row.addWidget(self.subfig_numbering_combo)
        self.subfig_numbering_combo.currentIndexChanged.connect(self._on_subfig_numbering_changed)
        # --- Detect Subfigures button ---
        self.detect_subfigures_btn = QPushButton("Detect Subfigures")
        self.detect_subfigures_btn.setToolTip("Detect subfigures in the selected figure")
        self.detect_subfigures_btn.clicked.connect(self.gallery_widget._on_detect_subfigures_clicked)
        btn_row.addWidget(self.detect_subfigures_btn)
        # --- Add a Subfigure button ---
        self.add_subfigure_btn = QPushButton("Add a Subfigure")
        self.add_subfigure_btn.clicked.connect(self._on_add_subfigure_clicked)
        btn_row.addWidget(self.add_subfigure_btn)
        self.save_subfigures_btn = QPushButton("Save Subfigures")
        self.save_subfigures_btn.setEnabled(False)  # Disabled by default
        self.save_subfigures_btn.clicked.connect(self.gallery_widget.save_subfigures)
        btn_row.addWidget(self.save_subfigures_btn)
        self.toggle_subfigures_btn = QPushButton("Hide Subfigures")
        def toggle_subfigures():
            self.show_db_subfigures = not self.show_db_subfigures
            if self.show_db_subfigures:
                self.toggle_subfigures_btn.setText("Hide Subfigures")
            else:
                self.toggle_subfigures_btn.setText("Show Subfigures")
            # Reload figures from DB with/without subfigures
            if self.main_window and hasattr(self.main_window, 'document_record'):
                self.show_figures_from_db(self.main_window.document_record)
        self.toggle_subfigures_btn.clicked.connect(toggle_subfigures)
        btn_row.addWidget(self.toggle_subfigures_btn)
        btn_row.addStretch(1)
        main_layout.addLayout(btn_row)
        gallery_widget_container = QWidget()
        gallery_widget_container.setLayout(main_layout)
        self.splitter.addWidget(gallery_widget_container)

        # Bottom: Table
        self.table_view = QTableView()
        self.table_view.setModel(self.model)
        self.table_view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table_view.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.splitter.addWidget(self.table_view)
        self.table_view.setColumnHidden(0, True)
        self.table_view.setColumnWidth(1, 100)
        self.table_view.setColumnWidth(2, 50)
        self.table_view.setColumnWidth(3, 70)
        self.table_view.setColumnWidth(4, 50)
        self.table_view.setColumnWidth(5, 40)
        self.table_view.setColumnWidth(6, 70)
        self.table_view.setColumnWidth(7, 50)
        self.table_view.setColumnWidth(8, 50)
        self.table_view.setColumnWidth(9, 300)
        self.table_view.horizontalHeader().setSectionResizeMode(len(self.model._headers) - 1, QHeaderView.ResizeMode.Stretch)
        #self.splitter.setStretchFactor(0, 3)
        #self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([300, 100])

        self.table_view.selectionModel().selectionChanged.connect(self._on_table_selection_changed)
        self.table_view.doubleClicked.connect(self._on_table_double_clicked)

        layout = QVBoxLayout(self)
        layout.addWidget(self.splitter)
        self.setLayout(layout)

        self.table_view.setStyleSheet('''
            QTableView::item:selected, QTableView QTableCornerButton::section:pressed {
                background: #2196F3;
                color: white;
            }
            QTableView::item:focus {
                outline: 2px solid #1976D2;
            }
            QHeaderView::section:selected {
                background: #2196F3;
                color: white;
            }
        ''')

    def _scroll_gallery_to_y(self, y):
        self.gallery_scroll.verticalScrollBar().setValue(y)

    def _on_table_selection_changed(self, selected, deselected):
        if getattr(self, '_syncing_selection', False):
            return
        self._syncing_selection = True
        try:
            if selected.indexes():
                row = selected.indexes()[0].row()
                fig = self.model.get_figure(row)
                is_subfigure = hasattr(fig, 'parent') and fig.parent is not None
                parent_idx = None
                subfig_idx = None
                if is_subfigure:
                    parent_id = fig.parent.id if hasattr(fig.parent, 'id') else fig.parent
                    for i, gfig in enumerate(self.gallery_widget.figures):
                        if gfig.get('id') == parent_id:
                            parent_idx = i
                            break
                    if parent_idx is not None and 'subfig_boxes' in self.gallery_widget.figures[parent_idx]:
                        try:
                            box = json.loads(fig.subfigure_box) if hasattr(fig, 'subfigure_box') and fig.subfigure_box else None
                            if box:
                                pixmap = self.gallery_widget.figures[parent_idx]['pixmap']
                                x = int(box['x'] * pixmap.width())
                                y = int(box['y'] * pixmap.height())
                                w = int(box['w'] * pixmap.width())
                                h = int(box['h'] * pixmap.height())
                                for idx, subfig in enumerate(self.gallery_widget.figures[parent_idx]['subfig_boxes']):
                                    rect = subfig['box']
                                    if (rect.x(), rect.y(), rect.width(), rect.height()) == (x, y, w, h):
                                        subfig_idx = idx
                                        break
                        except Exception:
                            pass
                    if parent_idx is not None:
                        self.gallery_widget.set_selected_index(parent_idx, subfig_idx)
                        self.gallery_widget.update()
                    fig_gal = self.gallery_widget.figures[parent_idx] if parent_idx is not None and 0 <= parent_idx < len(self.gallery_widget.figures) else None
                    if fig_gal and 'subfig_boxes' in fig_gal and fig_gal['subfig_boxes']:
                        self.save_subfigures_btn.setEnabled(True)
                    else:
                        self.save_subfigures_btn.setEnabled(False)
                else:
                    main_idx = None
                    for i, gfig in enumerate(self.gallery_widget.figures):
                        if gfig.get('id') == fig.id:
                            main_idx = i
                            break
                    if main_idx is not None:
                        if self.gallery_widget.selected_index != main_idx or self.gallery_widget._selected_subfig_box is not None:
                            self.gallery_widget.set_selected_index(main_idx, None)
                            self.gallery_widget.scroll_to_figure(main_idx)
                            self.gallery_widget.update()
                    self.save_subfigures_btn.setEnabled(False)
            else:
                if self.gallery_widget.selected_index is not None:
                    self.gallery_widget.set_selected_index(None)
                self.save_subfigures_btn.setEnabled(False)
        finally:
            self._syncing_selection = False

    def _on_table_double_clicked(self, index):
        row = index.row()
        fig = self.model.get_figure(row)
        if not fig or not self.main_window or not hasattr(self.main_window, 'document_record'):
            return
        dialog = FigureInfoDialog(fig, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.show_figures_from_db(self.main_window.document_record)

    def show_figures_from_db(self, document):
        # Filter figures based on show_db_subfigures
        def sort_key(fig):
            # Part 1: try numeric if possible
            p1 = int(fig.part1_number) if fig.part1_number and str(fig.part1_number).isdigit() else fig.part1_number

            # Part 2: sort NULLs last (use a string that sorts after all letters, e.g., 'ZZZ')
            p2 = fig.part2_number
            if p2 is None:
                return (p1, '')  # 'ZZZ' will sort after 'A', 'B', etc.
            elif str(p2).isdigit():
                return (p1, f"{int(p2):03d}")  # pad numbers for correct string sorting
            else:
                return (p1, str(p2))  # string sort (e.g., 'a', 'b')

        if getattr(self, 'show_db_subfigures', True):
            figures = list(PrFigure.select().where(PrFigure.document == document))
        else:
            figures = list(PrFigure.select().where((PrFigure.document == document) & (PrFigure.parent.is_null()) ))
        figures = sorted(figures, key=sort_key)
        self.model.set_figures(figures)
        # --- Detect subfigure numbering scheme from part2_number ---
        part2_numbers = [getattr(f, 'part2_number', None) for f in figures if getattr(f, 'part2_number', None)]
        scheme = '1,2,3'
        combo_idx = 0
        if part2_numbers:
            if all(str(n).isdigit() for n in part2_numbers):
                scheme = '1,2,3'
                combo_idx = 0
            elif all(isinstance(n, str) and n.isupper() and n.isalpha() for n in part2_numbers):
                scheme = 'A,B,C'
                combo_idx = 1
            elif all(isinstance(n, str) and n.islower() and n.isalpha() for n in part2_numbers):
                scheme = 'a,b,c'
                combo_idx = 2
        self.subfig_numbering_combo.setCurrentIndex(combo_idx)
        self.gallery_widget.set_subfig_numbering_scheme(scheme)
        # ---
        gallery_figures = []
        for fig in figures:
            pixmap = QPixmap()
            if fig.figure_binary:
                pixmap.loadFromData(fig.figure_binary)
            gallery_figures.append({'id': fig.id, 'pixmap': pixmap, 'caption': fig.caption_text or "", 
                                    'parent_id': fig.parent.id if fig.parent else None, 'subfigure_box': fig.subfigure_box})
        self.gallery_widget.set_figures(gallery_figures)

    def _on_gallery_selection_changed(self, idx, subfig_idx=None):
        if getattr(self, '_syncing_selection', False):
            return
        self._syncing_selection = True
        try:
            sel_model = self.table_view.selectionModel()
            target_id = None
            if idx is not None and 0 <= idx < len(self.gallery_widget.figures):
                fig = self.gallery_widget.figures[idx]
                if subfig_idx is not None and 'subfig_boxes' in fig and 0 <= subfig_idx < len(fig['subfig_boxes']):
                    subfig = fig['subfig_boxes'][subfig_idx]
                    target_id = subfig.get('id')
                else:
                    target_id = fig.get('id')
            row_to_select = None
            for row in range(self.model.rowCount()):
                model_fig = self.model.get_figure(row)
                if hasattr(model_fig, 'id') and model_fig.id == target_id:
                    row_to_select = row
                    break
            if row_to_select is not None:
                if sel_model:
                    index = self.model.index(row_to_select, 0)
                    if not sel_model.isSelected(index):
                        sel_model.clearSelection()
                        sel_model.select(index, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
                        self.table_view.scrollTo(index)
                fig = self.gallery_widget.figures[idx] if idx is not None and 0 <= idx < len(self.gallery_widget.figures) else None
                if fig and 'subfig_boxes' in fig and fig['subfig_boxes']:
                    self.save_subfigures_btn.setEnabled(True)
                else:
                    self.save_subfigures_btn.setEnabled(False)
            else:
                if sel_model and sel_model.hasSelection():
                    sel_model.clearSelection()
            fig = self.gallery_widget.figures[idx] if idx is not None and 0 <= idx < len(self.gallery_widget.figures) else None
            if fig and 'subfig_boxes' in fig and fig['subfig_boxes']:
                self.save_subfigures_btn.setEnabled(True)
            else:
                self.save_subfigures_btn.setEnabled(False)
        finally:
            self._syncing_selection = False

    def _on_subfigures_changed(self):
        # Called when subfigures are detected/changed
        idx = self.gallery_widget.selected_index
        fig = self.gallery_widget.figures[idx] if idx is not None and 0 <= idx < len(self.gallery_widget.figures) else None
        if fig and 'subfig_boxes' in fig and fig['subfig_boxes']:
            self.save_subfigures_btn.setEnabled(True)
        else:
            self.save_subfigures_btn.setEnabled(False)

    def _on_add_subfigure_clicked(self):
        # Enable add subfigure mode in gallery
        self.gallery_widget.add_subfigure_mode = True
        self.gallery_widget._add_subfig_start = None
        self.gallery_widget._add_subfig_end = None
        self.gallery_widget.update()

    def _on_subfig_numbering_changed(self, idx):
        # Update the numbering scheme and propagate to gallery widget
        if idx == 0:
            self.subfig_numbering_scheme = '1,2,3'
        elif idx == 1:
            self.subfig_numbering_scheme = 'A,B,C'
        elif idx == 2:
            self.subfig_numbering_scheme = 'a,b,c'
        else:
            self.subfig_numbering_scheme = '1,2,3'
        self.gallery_widget.set_subfig_numbering_scheme(self.subfig_numbering_scheme)

class SubfigureView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Create layout
        layout = QVBoxLayout()

