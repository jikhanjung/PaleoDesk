import os
from PyPDF2 import PdfReader, PdfWriter
from PyQt6.QtCore import QThread, pyqtSignal
import fitz  # PyMuPDF


class PDFProcessor(QThread):
    """PDF 병합 및 크롭 처리를 위한 워커 스레드"""
    
    progress = pyqtSignal(int)  # 진행률 (0-100)
    statusUpdate = pyqtSignal(str)  # 상태 메시지
    finished = pyqtSignal(bool, str)  # 성공 여부, 메시지
    
    def __init__(self, input_file, output_file, top_crop_percent=0, bottom_crop_percent=0, page_gap_percent=5):
        super().__init__()
        self.input_file = input_file
        self.output_file = output_file
        self.top_crop_percent = top_crop_percent
        self.bottom_crop_percent = bottom_crop_percent
        self.page_gap_percent = page_gap_percent
        
    def run(self):
        """PDF 처리 실행"""
        try:
            self.statusUpdate.emit('PDF 파일을 읽는 중...')
            
            # PyMuPDF로 PDF 열기
            pdf_document = fitz.open(self.input_file)
            total_pages = len(pdf_document)
            
            if total_pages == 0:
                self.finished.emit(False, 'PDF 파일에 페이지가 없습니다.')
                return
                
            # 새 PDF 문서 생성
            output_pdf = fitz.open()
            
            # 2페이지씩 처리
            pages_processed = 0
            for i in range(0, total_pages, 2):
                self.statusUpdate.emit(f'페이지 {i+1}-{min(i+2, total_pages)} 처리 중...')
                
                # 첫 번째 페이지
                page1 = pdf_document[i]
                rect1 = page1.rect
                
                # 크롭 적용 (퍼센티지를 픽셀로 변환)
                if self.top_crop_percent > 0 or self.bottom_crop_percent > 0:
                    page_height = rect1.height
                    top_crop_pixels = (self.top_crop_percent / 100.0) * page_height
                    bottom_crop_pixels = (self.bottom_crop_percent / 100.0) * page_height
                    
                    crop_rect1 = fitz.Rect(
                        rect1.x0,
                        rect1.y0 + top_crop_pixels,
                        rect1.x1,
                        rect1.y1 - bottom_crop_pixels
                    )
                    page1.set_cropbox(crop_rect1)
                    rect1 = crop_rect1
                
                # 두 번째 페이지 (있는 경우)
                if i + 1 < total_pages:
                    page2 = pdf_document[i + 1]
                    rect2 = page2.rect
                    
                    # 크롭 적용 (퍼센티지를 픽셀로 변환)
                    if self.top_crop_percent > 0 or self.bottom_crop_percent > 0:
                        page_height = rect2.height
                        top_crop_pixels = (self.top_crop_percent / 100.0) * page_height
                        bottom_crop_pixels = (self.bottom_crop_percent / 100.0) * page_height
                        
                        crop_rect2 = fitz.Rect(
                            rect2.x0,
                            rect2.y0 + top_crop_pixels,
                            rect2.x1,
                            rect2.y1 - bottom_crop_pixels
                        )
                        page2.set_cropbox(crop_rect2)
                        rect2 = crop_rect2
                else:
                    page2 = None
                    rect2 = rect1  # 빈 페이지를 위한 크기
                
                # 새 페이지 크기 계산 (두 페이지를 나란히)
                # 페이지 간격을 첫 번째 페이지 폭의 퍼센티지로 계산
                gap_pixels = (self.page_gap_percent / 100.0) * rect1.width
                new_width = rect1.width + (rect2.width if page2 else rect1.width) + gap_pixels
                new_height = max(rect1.height, rect2.height if page2 else rect1.height)
                
                # 새 페이지 생성
                new_page = output_pdf.new_page(width=new_width, height=new_height)
                
                # 첫 번째 페이지 그리기
                new_page.show_pdf_page(
                    fitz.Rect(0, 0, rect1.width, rect1.height),
                    pdf_document,
                    i
                )
                
                # 두 번째 페이지 그리기 (있는 경우)
                if page2:
                    new_page.show_pdf_page(
                        fitz.Rect(rect1.width + gap_pixels, 0, rect1.width + gap_pixels + rect2.width, rect2.height),
                        pdf_document,
                        i + 1
                    )
                
                pages_processed += 2
                progress = int((pages_processed / total_pages) * 100)
                self.progress.emit(progress)
                
            # PDF 저장
            self.statusUpdate.emit('PDF 파일을 저장하는 중...')
            output_pdf.save(self.output_file)
            
            # 정리
            output_pdf.close()
            pdf_document.close()
            
            self.finished.emit(True, f'PDF 병합 완료!\n저장 위치: {self.output_file}')
            
        except Exception as e:
            self.finished.emit(False, f'오류 발생: {str(e)}')


class PDFMerger:
    """PDF 병합 유틸리티 클래스"""
    
    @staticmethod
    def generate_output_filename(input_file):
        """출력 파일명 생성"""
        dir_path = os.path.dirname(input_file)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_name = f"{base_name}_merged.pdf"
        output_path = os.path.join(dir_path, output_name)
        
        # 파일이 이미 존재하면 번호 추가
        counter = 1
        while os.path.exists(output_path):
            output_name = f"{base_name}_merged_{counter}.pdf"
            output_path = os.path.join(dir_path, output_name)
            counter += 1
            
        return output_path