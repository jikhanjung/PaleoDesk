from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QSlider, QSpinBox, QPushButton, QGroupBox,
                             QDialogButtonBox)
from PyQt6.QtCore import Qt, pyqtSignal


class CropDialog(QDialog):
    """크롭 설정 다이얼로그"""
    
    cropChanged = pyqtSignal(int, int)  # top_crop, bottom_crop
    
    def __init__(self, parent=None, top_crop=0, bottom_crop=0):
        super().__init__(parent)
        self.top_crop = top_crop
        self.bottom_crop = bottom_crop
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('크롭 여백 설정')
        self.setModal(True)
        self.setFixedSize(400, 300)
        
        layout = QVBoxLayout(self)
        
        # 설명 레이블
        info_label = QLabel('PDF 페이지의 상하 여백을 설정합니다.\n값은 픽셀 단위입니다.')
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # 상단 크롭 설정
        top_group = QGroupBox('상단 여백')
        top_layout = QHBoxLayout(top_group)
        
        self.top_slider = QSlider(Qt.Orientation.Horizontal)
        self.top_slider.setMinimum(0)
        self.top_slider.setMaximum(500)
        self.top_slider.setValue(self.top_crop)
        self.top_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.top_slider.setTickInterval(50)
        
        self.top_spinbox = QSpinBox()
        self.top_spinbox.setMinimum(0)
        self.top_spinbox.setMaximum(500)
        self.top_spinbox.setValue(self.top_crop)
        self.top_spinbox.setSuffix(' px')
        
        # 연결
        self.top_slider.valueChanged.connect(self.top_spinbox.setValue)
        self.top_spinbox.valueChanged.connect(self.top_slider.setValue)
        self.top_slider.valueChanged.connect(self.onCropChanged)
        
        top_layout.addWidget(QLabel('크롭:'))
        top_layout.addWidget(self.top_slider, 1)
        top_layout.addWidget(self.top_spinbox)
        
        layout.addWidget(top_group)
        
        # 하단 크롭 설정
        bottom_group = QGroupBox('하단 여백')
        bottom_layout = QHBoxLayout(bottom_group)
        
        self.bottom_slider = QSlider(Qt.Orientation.Horizontal)
        self.bottom_slider.setMinimum(0)
        self.bottom_slider.setMaximum(500)
        self.bottom_slider.setValue(self.bottom_crop)
        self.bottom_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.bottom_slider.setTickInterval(50)
        
        self.bottom_spinbox = QSpinBox()
        self.bottom_spinbox.setMinimum(0)
        self.bottom_spinbox.setMaximum(500)
        self.bottom_spinbox.setValue(self.bottom_crop)
        self.bottom_spinbox.setSuffix(' px')
        
        # 연결
        self.bottom_slider.valueChanged.connect(self.bottom_spinbox.setValue)
        self.bottom_spinbox.valueChanged.connect(self.bottom_slider.setValue)
        self.bottom_slider.valueChanged.connect(self.onCropChanged)
        
        bottom_layout.addWidget(QLabel('크롭:'))
        bottom_layout.addWidget(self.bottom_slider, 1)
        bottom_layout.addWidget(self.bottom_spinbox)
        
        layout.addWidget(bottom_group)
        
        # 기본값 버튼
        default_btn = QPushButton('기본값으로 재설정')
        default_btn.clicked.connect(self.resetToDefault)
        layout.addWidget(default_btn)
        
        # 버튼
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
    def onCropChanged(self):
        """크롭 값 변경 시 시그널 발생"""
        self.cropChanged.emit(self.top_slider.value(), self.bottom_slider.value())
        
    def resetToDefault(self):
        """기본값으로 재설정"""
        self.top_slider.setValue(0)
        self.bottom_slider.setValue(0)
        
    def getCropValues(self):
        """현재 크롭 값 반환"""
        return self.top_slider.value(), self.bottom_slider.value()