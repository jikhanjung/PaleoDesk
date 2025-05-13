import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import tempfile
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PDFRefinery import resource_path, MainWindow

class TestPDFRefinery(unittest.TestCase):
    def setUp(self):
        # Store original sys._MEIPASS if it exists
        self.original_meipass = getattr(sys, '_MEIPASS', None)
        
        # Create QApplication instance for GUI tests
        self.app = QApplication(sys.argv)
        
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        # Restore original sys._MEIPASS
        if self.original_meipass is None:
            if hasattr(sys, '_MEIPASS'):
                delattr(sys, '_MEIPASS')
        else:
            sys._MEIPASS = self.original_meipass
            
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.test_dir)
        
        # Clean up QApplication
        self.app.quit()

    def test_resource_path_normal(self):
        """Test resource_path when running as normal Python script"""
        test_path = "test_file.txt"
        result = resource_path(test_path)
        expected = os.path.abspath(os.path.join(os.path.dirname(__file__), test_path))
        self.assertEqual(result, expected)

    def test_resource_path_pyinstaller(self):
        """Test resource_path when running as PyInstaller bundle"""
        # Simulate PyInstaller environment
        sys._MEIPASS = "/fake/pyinstaller/path"
        test_path = "test_file.txt"
        result = resource_path(test_path)
        expected = os.path.abspath(os.path.join(sys._MEIPASS, test_path))
        self.assertEqual(result, expected)

    def create_mock_pdf(self):
        """Create a mock PDF file for testing"""
        import fitz  # PyMuPDF
        pdf_path = os.path.join(self.test_dir, "test.pdf")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Test PDF Content")
        doc.save(pdf_path)
        doc.close()
        return pdf_path

    def test_load_pdf_file_success(self):
        """Test successful PDF file loading"""
        # Create main window instance
        main_window = MainWindow()
        
        # Create and load mock PDF
        pdf_path = self.create_mock_pdf()
        main_window.load_pdf_file(pdf_path)
        
        # Verify PDF was loaded
        self.assertIsNotNone(main_window.pdf_viewer)
        self.assertTrue(os.path.exists(pdf_path))
        
        # Clean up
        main_window.close()

    def test_load_pdf_file_invalid(self):
        """Test loading invalid PDF file"""
        # Create main window instance
        main_window = MainWindow()
        
        # Try to load non-existent file
        invalid_path = os.path.join(self.test_dir, "nonexistent.pdf")
        main_window.load_pdf_file(invalid_path)
        
        # Verify error handling
        self.assertIsNone(main_window.pdf_viewer)
        
        # Clean up
        main_window.close()

    def test_load_pdf_file_empty(self):
        """Test loading empty PDF file"""
        # Create main window instance
        main_window = MainWindow()
        
        # Create empty PDF
        pdf_path = os.path.join(self.test_dir, "empty.pdf")
        with open(pdf_path, 'wb') as f:
            f.write(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n')
        
        main_window.load_pdf_file(pdf_path)
        
        # Verify error handling
        self.assertIsNone(main_window.pdf_viewer)
        
        # Clean up
        main_window.close()

if __name__ == '__main__':
    unittest.main() 