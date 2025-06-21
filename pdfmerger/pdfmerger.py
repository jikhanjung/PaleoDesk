import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSplitter, QListWidget, QPushButton,
                             QLabel, QFileDialog, QListWidgetItem, QAbstractItemView,
                             QToolBar, QMessageBox, QProgressDialog)
from PyQt6.QtCore import Qt, QDir, QSize, QSettings
from PyQt6.QtGui import QAction, QIcon, QFont

from pdf_viewer import PDFViewer
from pdf_processor import PDFProcessor, PDFMerger


class PDFMergerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_directory = ""
        self.selected_pdf = None
        self.top_crop = 0
        self.bottom_crop = 0
        self.page_gap = 5  # 기본값 5%
        self.settings = QSettings('PaleoDesk', 'PDFMerger')
        self.loadSettings()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('PDF 악보 병합기')
        self.setGeometry(100, 100, 1200, 800)
        
        # 메인 위젯과 레이아웃
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # 스플리터로 좌우 패널 분할
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # 왼쪽 패널: 파일 브라우저
        left_panel = self.createFilePanel()
        splitter.addWidget(left_panel)
        
        # 오른쪽 패널: PDF 뷰어 (placeholder)
        right_panel = self.createViewerPanel()
        splitter.addWidget(right_panel)
        
        # 스플리터 비율 설정 (30:70)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)
        
        # 메뉴바 생성
        self.createMenuBar()
        
        # 툴바 생성
        self.createToolBar()
        
        # 상태바
        self.statusBar().showMessage('준비')
        
    def createFilePanel(self):
        """파일 브라우저 패널 생성"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(5)
        
        # 디렉토리 선택 섹션
        dir_section = QWidget()
        dir_layout = QVBoxLayout(dir_section)
        dir_layout.setContentsMargins(0, 0, 0, 0)
        
        # 레이블
        label = QLabel('PDF 파일 위치:')
        label.setFont(QFont("", 9, QFont.Weight.Bold))
        dir_layout.addWidget(label)
        
        # 경로 표시 및 버튼
        path_layout = QHBoxLayout()
        
        self.dir_label = QLabel('폴더를 선택하세요')
        self.dir_label.setStyleSheet("""
            QLabel { 
                padding: 8px; 
                background-color: #f5f5f5; 
                border: 1px solid #ddd;
                border-radius: 4px;
            }
        """)
        self.dir_label.setWordWrap(True)
        path_layout.addWidget(self.dir_label, 1)
        
        self.browse_btn = QPushButton('📁 폴더 선택')
        self.browse_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 15px;
                background-color: #0084ff;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0066cc;
            }
        """)
        self.browse_btn.clicked.connect(self.selectDirectory)
        path_layout.addWidget(self.browse_btn)
        
        dir_layout.addLayout(path_layout)
        layout.addWidget(dir_section)
        
        # 구분선
        line = QWidget()
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: #ddd;")
        layout.addWidget(line)
        
        # PDF 파일 목록 레이블
        list_label = QLabel('PDF 파일 목록:')
        list_label.setFont(QFont("", 9, QFont.Weight.Bold))
        layout.addWidget(list_label)
        
        # PDF 파일 목록
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.file_list.itemClicked.connect(self.onFileSelected)
        self.file_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #f0f0f0;
            }
            QListWidget::item:selected {
                background-color: #0084ff;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #f0f0f0;
            }
        """)
        layout.addWidget(self.file_list)
        
        # 파일 개수 표시
        self.file_count_label = QLabel('파일 없음')
        self.file_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_count_label.setStyleSheet("color: #666; padding: 5px;")
        layout.addWidget(self.file_count_label)
        
        return panel
        
    def createViewerPanel(self):
        """PDF 뷰어 패널 생성"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # PDF 뷰어 위젯 생성
        self.pdf_viewer = PDFViewer()
        self.pdf_viewer.pageChanged.connect(self.onPageChanged)
        self.pdf_viewer.cropChanged.connect(self.onCropChanged)
        self.pdf_viewer.gapChanged.connect(self.onGapChanged)
        layout.addWidget(self.pdf_viewer)
        
        # 병합 실행 버튼
        self.merge_button = QPushButton('🔄 PDF 병합 실행')
        self.merge_button.setMinimumHeight(60)
        self.merge_button.setEnabled(False)
        self.merge_button.clicked.connect(self.processPDF)
        self.merge_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
            QPushButton:disabled {
                background-color: #6c757d;
                color: #adb5bd;
            }
        """)
        layout.addWidget(self.merge_button)
        
        return panel
        
    def createMenuBar(self):
        """메뉴바 생성"""
        menubar = self.menuBar()
        
        # 파일 메뉴
        file_menu = menubar.addMenu('파일')
        
        open_action = QAction('폴더 열기', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.selectDirectory)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('종료', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 도움말 메뉴
        help_menu = menubar.addMenu('도움말')
        
        shortcuts_action = QAction('키보드 단축키', self)
        shortcuts_action.triggered.connect(self.showShortcuts)
        help_menu.addAction(shortcuts_action)
        
    def createToolBar(self):
        """툴바 생성"""
        toolbar = self.addToolBar('Main')
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        
        # 폴더 열기 액션
        open_action = QAction('📁\n폴더 열기', self)
        open_action.setStatusTip('PDF 파일이 있는 폴더를 선택합니다')
        open_action.triggered.connect(self.selectDirectory)
        toolbar.addAction(open_action)
        
        toolbar.addSeparator()
        
        # PDF 처리 액션
        self.process_action = QAction('⚙️\nPDF 병합', self)
        self.process_action.setStatusTip('선택한 PDF를 병합합니다')
        self.process_action.setEnabled(False)
        self.process_action.triggered.connect(self.processPDF)
        toolbar.addAction(self.process_action)
        
        
    def selectDirectory(self):
        """디렉토리 선택 다이얼로그"""
        directory = QFileDialog.getExistingDirectory(
            self,
            '폴더 선택',
            self.current_directory if self.current_directory else QDir.homePath()
        )
        
        if directory:
            self.current_directory = directory
            self.dir_label.setText(directory)
            self.loadPDFFiles()
            
    def loadPDFFiles(self):
        """선택된 디렉토리에서 PDF 파일 목록 로드"""
        self.file_list.clear()
        
        if not os.path.exists(self.current_directory):
            return
            
        # PDF 파일 필터링
        pdf_files = []
        for filename in os.listdir(self.current_directory):
            if filename.lower().endswith('.pdf'):
                pdf_files.append(filename)
                
        # 파일명 정렬
        pdf_files.sort()
        
        # 리스트에 추가
        for pdf_file in pdf_files:
            item = QListWidgetItem(f"📄 {pdf_file}")
            self.file_list.addItem(item)
            
        # 파일 개수 업데이트
        if pdf_files:
            self.file_count_label.setText(f'총 {len(pdf_files)}개 파일')
            self.file_count_label.setStyleSheet("color: #0084ff; padding: 5px; font-weight: bold;")
        else:
            self.file_count_label.setText('PDF 파일 없음')
            self.file_count_label.setStyleSheet("color: #666; padding: 5px;")
            
        # 상태바 업데이트
        self.statusBar().showMessage(f'{len(pdf_files)}개의 PDF 파일을 찾았습니다')
        
    def onFileSelected(self, item):
        """파일 선택 이벤트 처리"""
        # 아이콘 제거하고 실제 파일명 추출
        filename = item.text().replace('📄 ', '')
        filepath = os.path.join(self.current_directory, filename)
        
        # PDF 로드
        if self.pdf_viewer.loadPDF(filepath):
            self.selected_pdf = filepath
            # 크롭 값 및 간격 적용
            self.pdf_viewer.setCropValues(self.top_crop, self.bottom_crop)
            self.pdf_viewer.gap_slider.setValue(self.page_gap)
            self.pdf_viewer.page_gap = self.page_gap
            self.pdf_viewer.gap_label.setText(f'{self.page_gap}%')
            self.statusBar().showMessage(f'파일 로드됨: {filename}')
            
            # PDF 처리 버튼들 활성화
            self.process_action.setEnabled(True)
            self.merge_button.setEnabled(True)
        else:
            QMessageBox.warning(self, '오류', f'PDF 파일을 열 수 없습니다:\n{filename}')
            
    def onPageChanged(self, current_page, total_pages):
        """페이지 변경 이벤트 처리"""
        if total_pages > 0:
            display_page = current_page + 1
            self.statusBar().showMessage(f'페이지: {display_page}-{min(display_page + 1, total_pages)} / {total_pages}')
            
    def showShortcuts(self):
        """키보드 단축키 안내 다이얼로그"""
        shortcuts_text = """
<h3>키보드 단축키</h3>
<table>
<tr><td><b>←, PageUp</b></td><td>이전 페이지</td></tr>
<tr><td><b>→, PageDown</b></td><td>다음 페이지</td></tr>
<tr><td><b>Home</b></td><td>첫 페이지</td></tr>
<tr><td><b>End</b></td><td>마지막 페이지</td></tr>
<tr><td><b>F</b></td><td>화면 맞춤</td></tr>
<tr><td><b>Ctrl+O</b></td><td>폴더 열기</td></tr>
<tr><td><b>Ctrl+Q</b></td><td>종료</td></tr>
</table>
"""
        QMessageBox.information(self, '키보드 단축키', shortcuts_text)
        
    def onCropChanged(self, top, bottom):
        """크롭 값 변경 이벤트 처리"""
        self.top_crop = top
        self.bottom_crop = bottom
        self.saveSettings()  # 크롭 값 변경 시 자동 저장
        
    def onGapChanged(self, gap):
        """페이지 간격 변경 이벤트 처리"""
        self.page_gap = gap
        self.saveSettings()  # 간격 값 변경 시 자동 저장
        
            
    def processPDF(self):
        """PDF 병합 처리"""
        if not self.selected_pdf:
            QMessageBox.warning(self, '오류', 'PDF 파일을 먼저 선택하세요.')
            return
            
        # 출력 파일명 생성
        output_file = PDFMerger.generate_output_filename(self.selected_pdf)
        
        # 진행 다이얼로그 생성
        progress_dialog = QProgressDialog('PDF 병합 중...', '취소', 0, 100, self)
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setMinimumDuration(0)
        
        # PDF 프로세서 시작
        self.processor = PDFProcessor(
            self.selected_pdf, 
            output_file, 
            top_crop_percent=self.top_crop, 
            bottom_crop_percent=self.bottom_crop,
            page_gap_percent=self.page_gap
        )
        
        # 시그널 연결
        self.processor.progress.connect(progress_dialog.setValue)
        self.processor.statusUpdate.connect(progress_dialog.setLabelText)
        self.processor.finished.connect(self.onProcessFinished)
        progress_dialog.canceled.connect(self.processor.terminate)
        
        self.processor.start()
        progress_dialog.show()
        
    def onProcessFinished(self, success, message):
        """PDF 처리 완료"""
        if success:
            QMessageBox.information(self, '완료', message)
        else:
            QMessageBox.critical(self, '오류', message)
            
    def loadSettings(self):
        """설정 불러오기"""
        self.current_directory = self.settings.value('last_directory', '')
        self.top_crop = int(self.settings.value('top_crop', 0))
        self.bottom_crop = int(self.settings.value('bottom_crop', 0))
        self.page_gap = int(self.settings.value('page_gap', 5))
        
        # 윈도우 크기 및 위치
        geometry = self.settings.value('geometry')
        if geometry:
            self.restoreGeometry(geometry)
            
    def saveSettings(self):
        """설정 저장"""
        self.settings.setValue('last_directory', self.current_directory)
        self.settings.setValue('top_crop', self.top_crop)
        self.settings.setValue('bottom_crop', self.bottom_crop)
        self.settings.setValue('page_gap', self.page_gap)
        self.settings.setValue('geometry', self.saveGeometry())
        
    def closeEvent(self, event):
        """애플리케이션 종료 시 설정 저장"""
        self.saveSettings()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('PDF Merger')
    app.setOrganizationName('PaleoDesk')
    
    # 애플리케이션 스타일 설정 (선택사항)
    app.setStyle('Fusion')
    
    window = PDFMergerWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()