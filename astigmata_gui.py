import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QFrame, QScrollArea, 
                           QFileDialog, QSizePolicy, QProgressBar, QStackedWidget)
from PyQt5.QtCore import Qt, QMimeData, QTimer, QSize, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QDrag, QPixmap, QPainter, QPen, QColor, QFont, QPalette, QMovie

import torch
from torchvision import transforms
from PIL import Image
import random
from ModelGenerator import CNN  

class ImageDropZone(QLabel):
    imageDropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #4a90e2;
                border-radius: 10px;
                background-color: rgba(255, 255, 255, 0.1);
                padding: 20px;
            }
        """)
        self.pixmap = QPixmap(100, 100)
        self.pixmap.fill(Qt.transparent)
        self.setPixmap(self.pixmap)
        self.setMinimumSize(300, 300)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and event.mimeData().urls()[0].toLocalFile().lower().endswith(('.jpg', '.jpeg', '.png')):
            event.acceptProposedAction()
            self.setStyleSheet("""
                QLabel {
                    border: 2px dashed #4a90e2;
                    border-radius: 10px;
                    background-color: rgba(255, 255, 255, 0.2);
                    padding: 20px;
                }
            """)
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #4a90e2;
                border-radius: 10px;
                background-color: rgba(255, 255, 255, 0.1);
                padding: 20px;
            }
        """)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            file_path = event.mimeData().urls()[0].toLocalFile()
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.pixmap.load(file_path)
                self.setPixmap(self.pixmap)
                self.imageDropped.emit(file_path)
                event.acceptProposedAction()
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #4a90e2;
                border-radius: 10px;
                background-color: rgba(255, 255, 255, 0.1);
                padding: 20px;
            }
        """)

class LoadingAnimation(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(60, 60)
        self.angle = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.rotate)
        self.timer.start(40)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(self.angle)
        pen = QPen(Qt.white, 2)
        painter.setPen(pen)
        for i in range(8):
            painter.drawLine(0, 10, 0, 25 if i % 2 == 0 else 20)
            painter.rotate(45)

    def rotate(self):
        self.angle = (self.angle + 5) % 360
        self.update()

    def showEvent(self, event):
        self.timer.start()

    def hideEvent(self, event):
        self.timer.stop()

class InferenceThread(QThread):
    resultsReady = pyqtSignal(int, float, list)

    def __init__(self, model, image, transform):
        super().__init__()
        self.model = model
        self.image = image
        self.transform = transform

    def run(self):
        # Preprocess image and run inference
        image = self.transform(self.image).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.softmax(outputs, dim=1).max().item()
        
        # Prepare symptoms
        AstigmatsmSymtoms = ["Blurred or distorted vision.", "Eyestrain or discomfort.", "Headaches"]
        CataractSymptoms = ["Clouded, blurred, or dim vision.", "Difficulty seeing at night.", "Sensitivity to light and glare.", "Seeing 'halos' around lights.", "Frequent changes in eyeglass or contact lens prescription.", "Fading or yellowing of colors.", "Double vision in a single eye."]
        DiabeticRetinopathySymptoms = ["Spots or dark strings (floaters) in vision.", "Blurred vision.", "Fluctuating vision.", "Impaired color vision.", "Dark or empty areas in your vision.", "Vision loss."]
        symptoms = []

        if predicted.item() == 0:
            symptoms = AstigmatsmSymtoms
        elif predicted.item() == 1:
            symptoms = CataractSymptoms
        elif predicted.item() == 2:
            symptoms = DiabeticRetinopathySymptoms
        result = (predicted.item(), confidence, symptoms)
        self.resultsReady.emit(*result)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Team Astigmata")
        self.setMinimumSize(900, 600)
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #2c2f36;
                color: white;
            }
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: 1px solid white;
                border-radius: 5px;
                padding: 8px 15px;
            }
            QPushButton:hover {
                background-color: #357ab7;
            }
            QProgressBar {
                border: 1px solid white;
                border-radius: 5px;
                text-align: center;
                color: white;
                background-color: #2b2b2b;
            }
            QProgressBar::chunk {
                background-color: #ff5555;
                width: 10px;
                margin: 1px;
            }
        """)

        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        header_label = QLabel("Astigmatism Detection")
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setFont(QFont("Arial", 16, QFont.Bold))
        main_layout.addWidget(header_label)

        content_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_label = QLabel("Drop or Browse an Eye Image")
        left_label.setFont(QFont("Arial", 12))
        left_layout.addWidget(left_label)

        self.drop_zone = ImageDropZone()
        self.drop_zone.imageDropped.connect(self.process_image)
        left_layout.addWidget(self.drop_zone)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_image)
        left_layout.addWidget(browse_button, 0, Qt.AlignCenter)

        self.notice_label = QLabel("This result should not be\nconsidered a medical diagnosis.")
        self.notice_label.setFont(QFont("Arial", 10))
        left_layout.addWidget(self.notice_label)
        left_layout.addStretch()
        content_layout.addLayout(left_layout, 4)

        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setStyleSheet("background-color: white;")
        content_layout.addWidget(line)

        right_layout = QVBoxLayout()
        results_label = QLabel("Results")
        results_label.setFont(QFont("Arial", 12, QFont.Bold))
        right_layout.addWidget(results_label)

        self.loading_container = QWidget()
        loading_layout = QVBoxLayout(self.loading_container)
        self.loading_animation = LoadingAnimation()
        loading_layout.addWidget(self.loading_animation, 0, Qt.AlignCenter)
        self.loading_container.setVisible(False)
        right_layout.addWidget(self.loading_container)

        self.detection_label = QLabel()
        self.detection_label.setFont(QFont("Arial", 11))
        self.detection_label.setVisible(False)
        right_layout.addWidget(self.detection_label)

        self.infection_bar = QProgressBar()
        self.infection_bar.setVisible(False)
        self.infection_bar.setMaximum(100)
        self.infection_bar.setMinimum(0)
        right_layout.addWidget(self.infection_bar)

        self.symptoms_label = QLabel("Symptoms")
        self.symptoms_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.symptoms_label.setVisible(False)
        right_layout.addWidget(self.symptoms_label)

        self.symptoms_content = QWidget()
        self.symptoms_layout = QVBoxLayout(self.symptoms_content)
        self.symptoms_content.setStyleSheet("""
            QWidget {
                background-color: #222;
                border-radius: 5px;
                border: 1px solid white;
                padding: 10px;
            }
        """)
        self.symptoms_content.setVisible(False)
        right_layout.addWidget(self.symptoms_content)
        right_layout.addStretch()

        content_layout.addLayout(right_layout, 6)
        main_layout.addLayout(content_layout)
        self.setCentralWidget(central_widget)

        self.setWindowFlags(Qt.FramelessWindowHint)
        self._drag_pos = None

        self.model = CNN(num_classes=4)
        self.model.load_state_dict(torch.load("cnn_model.pth", map_location=torch.device('cpu')))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "JPEG Images (*.jpg *.jpeg *.png)")
        if file_path:
            self.process_image(file_path)

    def process_image(self, image_path):
        self.clear_results()
        self.drop_zone.pixmap.load(image_path)
        self.drop_zone.setPixmap(self.drop_zone.pixmap)
        self.loading_container.setVisible(True)
        self.notice_label.setVisible(True)

        image = Image.open(image_path).convert("RGB")

        # Start inference in background thread
        self.inference_thread = InferenceThread(self.model, image, self.transform)
        self.inference_thread.resultsReady.connect(self.display_results)
        self.inference_thread.start()

    def display_results(self, Class, confidence, symptoms):
        self.loading_container.setVisible(False)

        self.detection_label.setText(f"Condition Detected: {['Astigmatism', 'Cataract', 'Diabetic Retinopathy', 'Healthy'][Class]}")
        self.detection_label.setVisible(True)

        self.infection_bar.setValue(int(confidence * 100))
        self.infection_bar.setFormat(f"Infection Severity: {round(confidence * 100, 2)}%")
        self.infection_bar.setVisible(True)

        self.symptoms_label.setVisible(Class != 3)
        self.symptoms_content.setVisible(Class != 3)

        for i in reversed(range(self.symptoms_layout.count())):
            widget = self.symptoms_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        for symptom in symptoms:
            symptom_label = QLabel(f"• {symptom}")
            symptom_label.setFont(QFont("Arial", 10))
            symptom_label.setWordWrap(True)
            self.symptoms_layout.addWidget(symptom_label)

    def clear_results(self):
        self.loading_container.setVisible(False)
        self.detection_label.setVisible(False)
        self.infection_bar.setVisible(False)
        self.symptoms_label.setVisible(False)
        self.symptoms_content.setVisible(False)
        for i in reversed(range(self.symptoms_layout.count())):
            widget = self.symptoms_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
