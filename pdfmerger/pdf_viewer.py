import fitz  # PyMuPDF
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QScrollArea, QGraphicsView, QGraphicsScene,
                             QPushButton, QSlider, QSpinBox, QGraphicsRectItem)
from PyQt6.QtCore import Qt, pyqtSignal, QRectF
from PyQt6.QtGui import QPixmap, QImage, QPainter, QBrush, QColor, QKeyEvent, QPen


class PDFViewer(QWidget):
    """PDF 뷰어 위젯 - 두 페이지를 나란히 표시"""
    
    pageChanged = pyqtSignal(int, int)  # current_page, total_pages
    cropChanged = pyqtSignal(int, int)  # top_crop, bottom_crop
    gapChanged = pyqtSignal(int)  # page_gap
    
    def __init__(self):
        super().__init__()
        self.pdf_document = None
        self.current_page = 0
        self.total_pages = 0
        self.zoom_level = 1.0
        self.top_crop = 0
        self.bottom_crop = 0
        self.page_gap = 5  # 기본 간격 5% (페이지 폭의 5%)
        self.crop_overlays = []
        self.show_crop = True
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout(self)
        
        # 컨트롤 패널
        control_panel = self.createControlPanel()
        layout.addWidget(control_panel)
        
        # PDF 표시 영역
        self.graphics_view = QGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 스크롤바 정책
        self.graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        layout.addWidget(self.graphics_view)
        
    def createControlPanel(self):
        """컨트롤 패널 생성"""
        panel = QWidget()
        layout = QHBoxLayout(panel)
        
        # 이전 페이지 버튼
        self.prev_btn = QPushButton('◀ 이전')
        self.prev_btn.clicked.connect(self.previousPage)
        self.prev_btn.setEnabled(False)
        layout.addWidget(self.prev_btn)
        
        # 페이지 정보
        self.page_info = QLabel('페이지: 0 / 0')
        self.page_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.page_info.setMinimumWidth(150)
        layout.addWidget(self.page_info)
        
        # 다음 페이지 버튼
        self.next_btn = QPushButton('다음 ▶')
        self.next_btn.clicked.connect(self.nextPage)
        self.next_btn.setEnabled(False)
        layout.addWidget(self.next_btn)
        
        layout.addStretch()
        
        # 상단 크롭 설정
        layout.addWidget(QLabel('상단 크롭:'))
        self.top_crop_slider = QSlider(Qt.Orientation.Horizontal)
        self.top_crop_slider.setMinimum(0)
        self.top_crop_slider.setMaximum(50)  # 최대 50%
        self.top_crop_slider.setValue(0)
        self.top_crop_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.top_crop_slider.setTickInterval(10)
        self.top_crop_slider.valueChanged.connect(self.onTopCropChanged)
        self.top_crop_slider.setMinimumWidth(100)
        layout.addWidget(self.top_crop_slider)
        
        self.top_crop_label = QLabel('0%')
        self.top_crop_label.setMinimumWidth(40)
        layout.addWidget(self.top_crop_label)
        
        # 하단 크롭 설정
        layout.addWidget(QLabel('하단 크롭:'))
        self.bottom_crop_slider = QSlider(Qt.Orientation.Horizontal)
        self.bottom_crop_slider.setMinimum(0)
        self.bottom_crop_slider.setMaximum(50)  # 최대 50%
        self.bottom_crop_slider.setValue(0)
        self.bottom_crop_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.bottom_crop_slider.setTickInterval(10)
        self.bottom_crop_slider.valueChanged.connect(self.onBottomCropChanged)
        self.bottom_crop_slider.setMinimumWidth(100)
        layout.addWidget(self.bottom_crop_slider)
        
        self.bottom_crop_label = QLabel('0%')
        self.bottom_crop_label.setMinimumWidth(40)
        layout.addWidget(self.bottom_crop_label)
        
        # 페이지 간격 설정
        layout.addWidget(QLabel('페이지 간격:'))
        self.gap_slider = QSlider(Qt.Orientation.Horizontal)
        self.gap_slider.setMinimum(0)
        self.gap_slider.setMaximum(30)  # 최대 30% (페이지 폭의 30%)
        self.gap_slider.setValue(5)  # 기본값 5%
        self.gap_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.gap_slider.setTickInterval(5)
        self.gap_slider.valueChanged.connect(self.onGapChanged)
        self.gap_slider.setMinimumWidth(100)
        layout.addWidget(self.gap_slider)
        
        self.gap_label = QLabel('5%')
        self.gap_label.setMinimumWidth(40)
        layout.addWidget(self.gap_label)
        
        # 화면 맞춤 버튼
        self.fit_btn = QPushButton('🔍 화면 맞춤')
        self.fit_btn.clicked.connect(self.fitToWindow)
        layout.addWidget(self.fit_btn)
        
        # 크롭 표시 토글
        self.crop_toggle_btn = QPushButton('✂️ 크롭 표시')
        self.crop_toggle_btn.setCheckable(True)
        self.crop_toggle_btn.setChecked(True)
        self.crop_toggle_btn.toggled.connect(self.toggleCropDisplay)
        layout.addWidget(self.crop_toggle_btn)
        
        # 페이지 직접 입력
        layout.addWidget(QLabel('페이지:'))
        self.page_spinbox = QSpinBox()
        self.page_spinbox.setMinimum(1)
        self.page_spinbox.valueChanged.connect(self.onPageNumberChanged)
        layout.addWidget(self.page_spinbox)
        
        return panel
        
    def loadPDF(self, filepath):
        """PDF 파일 로드"""
        try:
            # 기존 문서 닫기
            if self.pdf_document:
                self.pdf_document.close()
                
            # 새 문서 열기
            self.pdf_document = fitz.open(filepath)
            self.total_pages = len(self.pdf_document)
            self.current_page = 0
            
            # UI 업데이트
            self.page_spinbox.setMaximum(self.total_pages - 1 if self.total_pages > 0 else 1)
            self.page_spinbox.setValue(1)
            
            # 첫 페이지 표시
            self.displayPages()
            
            return True
            
        except Exception as e:
            print(f"PDF 로드 오류: {e}")
            return False
            
    def displayPages(self):
        """현재 페이지와 다음 페이지를 나란히 표시"""
        if not self.pdf_document or self.total_pages == 0:
            return
            
        # 씬 초기화
        self.graphics_scene.clear()
        self.crop_overlays.clear()
        
        # 두 페이지 렌더링
        pixmaps = []
        page_heights = []
        
        # 첫 번째 페이지 (왼쪽)
        if self.current_page < self.total_pages:
            page1 = self.pdf_document[self.current_page]
            mat = fitz.Matrix(self.zoom_level, self.zoom_level)
            pix1 = page1.get_pixmap(matrix=mat)
            img1 = QImage(pix1.samples, pix1.width, pix1.height, pix1.stride, QImage.Format.Format_RGB888)
            pixmaps.append(QPixmap.fromImage(img1))
            page_heights.append(pix1.height)
        
        # 두 번째 페이지 (오른쪽)
        if self.current_page + 1 < self.total_pages:
            page2 = self.pdf_document[self.current_page + 1]
            mat = fitz.Matrix(self.zoom_level, self.zoom_level)
            pix2 = page2.get_pixmap(matrix=mat)
            img2 = QImage(pix2.samples, pix2.width, pix2.height, pix2.stride, QImage.Format.Format_RGB888)
            pixmaps.append(QPixmap.fromImage(img2))
            page_heights.append(pix2.height)
            
        # 페이지 배치
        x_offset = 0
        for i, pixmap in enumerate(pixmaps):
            self.graphics_scene.addPixmap(pixmap).setPos(x_offset, 0)
            
            # 크롭 오버레이 추가
            if self.show_crop and (self.top_crop > 0 or self.bottom_crop > 0):
                self.addCropOverlay(x_offset, 0, pixmap.width(), page_heights[i])
            
            # 페이지 간격을 픽셀로 변환 (페이지 폭의 퍼센티지)
            gap_pixels = (self.page_gap / 100.0) * pixmap.width()
            x_offset += pixmap.width() + gap_pixels
            
        # 마지막 페이지가 홀수인 경우 오른쪽에 빈 공간 추가 (일관된 레이아웃 유지)
        if len(pixmaps) == 1 and self.current_page + 1 >= self.total_pages:
            # 빈 페이지 영역 크기 (첫 번째 페이지와 동일한 크기)
            empty_width = pixmaps[0].width()
            empty_height = pixmaps[0].height()
            
            # 빈 영역을 시각적으로 표시 (선택사항 - 회색 테두리)
            empty_rect = QGraphicsRectItem(x_offset, 0, empty_width, empty_height)
            empty_rect.setPen(QPen(QColor(200, 200, 200), 1))  # 연한 회색 테두리
            empty_rect.setBrush(QBrush(QColor(250, 250, 250)))  # 매우 연한 회색 배경
            self.graphics_scene.addItem(empty_rect)
            
        # 씬 크기 조정
        self.graphics_scene.setSceneRect(self.graphics_scene.itemsBoundingRect())
        
        # 뷰 업데이트
        self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        
        # UI 업데이트
        self.updateUI()
        
    def updateUI(self):
        """UI 컨트롤 상태 업데이트"""
        # 페이지 정보
        display_page = self.current_page + 1
        self.page_info.setText(f'페이지: {display_page}-{min(display_page + 1, self.total_pages)} / {self.total_pages}')
        
        # 버튼 상태
        self.prev_btn.setEnabled(self.current_page > 0)
        self.next_btn.setEnabled(self.current_page + 2 < self.total_pages)
        
        # 페이지 번호 업데이트 (시그널 차단하여 무한 루프 방지)
        self.page_spinbox.blockSignals(True)
        self.page_spinbox.setValue(display_page)
        self.page_spinbox.blockSignals(False)
        
        # 시그널 발생
        self.pageChanged.emit(self.current_page, self.total_pages)
        
    def previousPage(self):
        """이전 페이지로 이동 (2페이지씩)"""
        if self.current_page >= 2:
            self.current_page -= 2
            self.displayPages()
            
    def nextPage(self):
        """다음 페이지로 이동 (2페이지씩)"""
        if self.current_page + 2 < self.total_pages:
            self.current_page += 2
            self.displayPages()
            
    def onPageNumberChanged(self, value):
        """페이지 번호 직접 입력"""
        # 홀수 페이지로 조정 (1, 3, 5...)
        page_num = value - 1
        if page_num % 2 == 1:
            page_num -= 1
        
        if 0 <= page_num < self.total_pages:
            self.current_page = page_num
            self.displayPages()
            
            
    def fitToWindow(self):
        """윈도우에 맞춤"""
        if self.graphics_scene.items():
            self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            
    def getPageSize(self):
        """현재 페이지 크기 반환"""
        if self.pdf_document and self.current_page < self.total_pages:
            page = self.pdf_document[self.current_page]
            return page.rect.width, page.rect.height
        return 0, 0
        
    def keyPressEvent(self, event: QKeyEvent):
        """키보드 단축키 처리"""
        if event.key() == Qt.Key.Key_Left or event.key() == Qt.Key.Key_PageUp:
            self.previousPage()
        elif event.key() == Qt.Key.Key_Right or event.key() == Qt.Key.Key_PageDown:
            self.nextPage()
        elif event.key() == Qt.Key.Key_Home:
            # 첫 페이지로
            self.current_page = 0
            self.displayPages()
        elif event.key() == Qt.Key.Key_End:
            # 마지막 페이지로 (짝수 페이지 맞춤)
            self.current_page = (self.total_pages // 2) * 2
            if self.current_page >= self.total_pages:
                self.current_page = self.total_pages - 2
            if self.current_page < 0:
                self.current_page = 0
            self.displayPages()
        elif event.key() == Qt.Key.Key_F:
            # 화면 맞춤
            self.fitToWindow()
        else:
            super().keyPressEvent(event)
            
    def addCropOverlay(self, x, y, width, height):
        """크롭 영역을 반투명 오버레이로 표시"""
        # 퍼센티지를 픽셀로 변환
        top_crop_pixels = (self.top_crop / 100.0) * height
        bottom_crop_pixels = (self.bottom_crop / 100.0) * height
        
        # 상단 크롭 영역
        if self.top_crop > 0:
            top_rect = QGraphicsRectItem(x, y, width, top_crop_pixels)
            top_rect.setBrush(QBrush(QColor(0, 0, 0, 128)))  # 반투명 검정
            top_rect.setPen(QPen(Qt.PenStyle.NoPen))
            self.graphics_scene.addItem(top_rect)
            self.crop_overlays.append(top_rect)
            
        # 하단 크롭 영역
        if self.bottom_crop > 0:
            bottom_y = y + height - bottom_crop_pixels
            bottom_rect = QGraphicsRectItem(x, bottom_y, width, bottom_crop_pixels)
            bottom_rect.setBrush(QBrush(QColor(0, 0, 0, 128)))  # 반투명 검정
            bottom_rect.setPen(QPen(Qt.PenStyle.NoPen))
            self.graphics_scene.addItem(bottom_rect)
            self.crop_overlays.append(bottom_rect)
            
    def setCropValues(self, top, bottom):
        """크롭 값 설정 (퍼센티지)"""
        self.top_crop = top
        self.bottom_crop = bottom
        
        # 슬라이드바 업데이트 (시그널 차단하여 무한 루프 방지)
        self.top_crop_slider.blockSignals(True)
        self.bottom_crop_slider.blockSignals(True)
        
        self.top_crop_slider.setValue(top)
        self.bottom_crop_slider.setValue(bottom)
        self.top_crop_label.setText(f'{top}%')
        self.bottom_crop_label.setText(f'{bottom}%')
        
        self.top_crop_slider.blockSignals(False)
        self.bottom_crop_slider.blockSignals(False)
        
        self.displayPages()
        self.cropChanged.emit(self.top_crop, self.bottom_crop)
        
    def toggleCropDisplay(self, checked):
        """크롭 표시 토글"""
        self.show_crop = checked
        self.displayPages()
        
    def onTopCropChanged(self, value):
        """상단 크롭 값 변경 (퍼센티지)"""
        self.top_crop = value
        self.top_crop_label.setText(f'{value}%')
        self.displayPages()
        self.cropChanged.emit(self.top_crop, self.bottom_crop)
        
    def onBottomCropChanged(self, value):
        """하단 크롭 값 변경 (퍼센티지)"""
        self.bottom_crop = value
        self.bottom_crop_label.setText(f'{value}%')
        self.displayPages()
        self.cropChanged.emit(self.top_crop, self.bottom_crop)
        
    def onGapChanged(self, value):
        """페이지 간격 값 변경 (퍼센티지)"""
        self.page_gap = value
        self.gap_label.setText(f'{value}%')
        self.displayPages()
        self.gapChanged.emit(self.page_gap)