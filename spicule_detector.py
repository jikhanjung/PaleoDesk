import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QWidget
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


def spicule_likelihood_from_lines(image_path, max_lines=300):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=25,
                            minLineLength=20, maxLineGap=5)
    num_lines = len(lines) if lines is not None else 0
    likelihood = min(num_lines / max_lines, 1.0)

    # Draw lines on a copy of the original image
    line_image = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 1)

    return likelihood, num_lines, line_image



class SpiculeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spicule Fossil Detector")
        self.setGeometry(100, 100, 800, 600)

        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)

        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)

        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.load_button)
        layout.addWidget(self.result_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            # make image fit the label size
            image = QImage(file_path)
            if image.isNull():
                self.image_label.setText("Failed to load image")
                return
            self.image_label.setPixmap(QPixmap.fromImage(image).scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            self.image_label.setText("")
            # analyze the image
            self.result_label.setText("Analyzing...")
            self.result_label.repaint()
            # run the analysis in a separate thread to avoid blocking the UI
            # (not implemented here for simplicity)
            # In a real application, you would use QThread or similar to run this in the background
            #self.show_image(file_path)
            # Call the analysis function
            self.analyze_image(file_path)

    def show_image(self, path):
        image = QImage(path)
        if image.isNull():
            self.image_label.setText("Failed to load image")
        else:
            self.image_label.setPixmap(QPixmap.fromImage(image))

    def analyze_image(self, path):
        try:
            score, count, line_image = spicule_likelihood_from_lines(path)

            # Convert OpenCV BGR to RGB for Qt
            rgb_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb_image.shape
            bytes_per_line = channels * width
            qimage = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Set image in QLabel
            self.image_label.setPixmap(QPixmap.fromImage(qimage).scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

            classification = "Yes ✅" if score > 0.6 else "No ❌"
            self.result_label.setText(
                f"Spicule fossil? {classification}\n"
                f"Detected lines: {count} | Likelihood: {score:.2f}"
            )
        except Exception as e:
            self.result_label.setText(f"Error: {str(e)}")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpiculeApp()
    window.show()
    sys.exit(app.exec_())
