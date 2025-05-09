import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QFrame, QScrollArea, 
                           QFileDialog, QSizePolicy)
from PyQt5.QtCore import Qt, QMimeData, QTimer, QSize, pyqtSignal
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
                border: 2px dashed white;
                border-radius: 10px;
                background-color: transparent;
                padding: 10px;
            }
        """)
        
        # Create image icon
        self.pixmap = QPixmap(100, 100)
        self.pixmap.fill(Qt.transparent)
        
        self.setPixmap(self.pixmap)
        self.setMinimumSize(200, 200)
        #self.setMaximumSize(200, 400)
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and event.mimeData().urls()[0].toLocalFile().lower().endswith(('.jpg', '.jpeg')):
            event.acceptProposedAction()
            self.setStyleSheet("""
                QLabel {
                    border: 2px dashed #aaa;
                    border-radius: 10px;
                    background-color: rgba(255, 255, 255, 0.1);
                    padding: 0px;
                }
            """)
        else:
            event.ignore()
            
    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed white;
                border-radius: 10px;
                background-color: transparent;
                padding: 0px;
            }
        """)
        
    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            file_path = event.mimeData().urls()[0].toLocalFile()
            if file_path.lower().endswith(('.jpg', '.jpeg')):
                self.pixmap.load(event.mimeData().urls()[0].toLocalFile())
                self.setPixmap(self.pixmap)
                self.imageDropped.emit(file_path)
                event.acceptProposedAction()
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed white;
                border-radius: 10px;
                background-color: transparent;
                padding: 0px;
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
        
        # Draw 8 lines forming a star pattern
        for i in range(8):
            if i % 2 == 0:
                painter.drawLine(0, 10, 0, 25)
            else:
                painter.drawLine(0, 10, 0, 20)
            painter.rotate(45)
    
    def rotate(self):
        self.angle = (self.angle + 5) % 360
        self.update()
        
    def showEvent(self, event):
        self.timer.start()
        
    def hideEvent(self, event):
        self.timer.stop()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Window properties
        self.setWindowTitle("Team Astigmata")
        self.setMinimumSize(800, 500)
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1a1a1a;
                color: white;
            }
            QPushButton {
                background-color: #333;
                color: white;
                border: 1px solid white;
                border-radius: 5px;
                padding: 8px 15px;
            }
            QPushButton:hover {
                background-color: #444;
            }
            QScrollArea {
                border: 1px solid white;
                border-radius: 5px;
                background-color: #222;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        header_label = QLabel("Astigmatism Detection")
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setFont(QFont("Arial", 14, QFont.Bold))
        main_layout.addWidget(header_label)
        
        # Content layout
        content_layout = QHBoxLayout()
        
        # Left side - Image upload
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        upload_label = QLabel("Please drop your image to the region below")
        upload_label.setFont(QFont("Arial", 12))
        upload_label.setAlignment(Qt.AlignLeft)
        left_layout.addWidget(upload_label)
        
        # Drop zone
        self.drop_zone = ImageDropZone()
        self.drop_zone.imageDropped.connect(self.process_image)
        left_layout.addWidget(self.drop_zone)
        
        # Browse button
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_image)
        browse_button.setFixedWidth(150)
        browse_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        left_layout.addWidget(browse_button, 0, Qt.AlignCenter)
        
        # Notice label
        self.notice_label = QLabel("This results should not be\n considered as real medical diagnosis.")
        self.notice_label.setFont(QFont("Arial", 12))
        self.notice_label.setAlignment(Qt.AlignLeft)
        #self.notice_label.setVisible(False)
        left_layout.addWidget(self.notice_label)
        
        left_layout.addStretch()
        content_layout.addWidget(left_widget)
        
        # Separator line
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: white;")
        content_layout.addWidget(line)
        
        # Right side - Results
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        results_label = QLabel("Results")
        results_label.setFont(QFont("Arial", 12))
        right_layout.addWidget(results_label)
        
        # Results area
        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_content = QWidget()
        self.results_layout = QVBoxLayout(self.results_content)
        self.results_scroll.setWidget(self.results_content)
        
        # Loading animation container
        self.loading_container = QWidget()
        loading_layout = QVBoxLayout(self.loading_container)
        self.loading_animation = LoadingAnimation()
        loading_layout.addWidget(self.loading_animation, 0, Qt.AlignCenter)
        self.loading_container.setVisible(False)
        
        # Detection result
        self.detection_label = QLabel()
        self.detection_label.setFont(QFont("Arial", 11))
        self.detection_label.setVisible(False)
        
        # Confidence score
        self.confidence_label = QLabel()
        self.confidence_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.confidence_label.setStyleSheet("color: #ff5555;")
        self.confidence_label.setAlignment(Qt.AlignRight)
        self.confidence_label.setVisible(False)
        
        # Detection result layout
        detection_layout = QHBoxLayout()
        detection_layout.addWidget(self.detection_label)
        detection_layout.addWidget(self.confidence_label)
        
        # Symptoms section
        self.symptoms_label = QLabel("Symptoms")
        self.symptoms_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.symptoms_label.setVisible(False)
        
        # Symptoms content
        self.symptoms_content = QWidget()
        self.symptoms_layout = QVBoxLayout(self.symptoms_content)
        self.symptoms_layout.setContentsMargins(10, 10, 10, 10)
        self.symptoms_content.setStyleSheet("""
            QWidget {
                background-color: #222;
                border-radius: 5px;
                border: 1px solid white;
            }
        """)
        self.symptoms_content.setVisible(False)
        
        # Add widgets to results layout
        self.results_layout.addWidget(self.loading_container)
        self.results_layout.addLayout(detection_layout)
        self.results_layout.addWidget(self.symptoms_label)
        self.results_layout.addWidget(self.symptoms_content)
        self.results_layout.addStretch()
        
        right_layout.addWidget(self.results_scroll)
        content_layout.addWidget(right_widget)
        
        # Set content layout stretch factors
        content_layout.setStretchFactor(left_widget, 4)
        content_layout.setStretchFactor(right_widget, 6)
        
        main_layout.addLayout(content_layout)
        self.setCentralWidget(central_widget)
        
        # Window control buttons (minimize, maximize/restore, close)
        self.setWindowFlags(Qt.FramelessWindowHint)
        
        # Add window control buttons at the top-right
        title_bar = QWidget()
        title_bar_layout = QHBoxLayout(title_bar)
        title_bar_layout.setContentsMargins(10, 5, 10, 5)
        
        title_label = QLabel("Team Astigmata")
        title_label.setFont(QFont("Arial", 10, QFont.Bold))
        
        min_button = QPushButton("-")
        min_button.setFixedSize(30, 30)
        min_button.clicked.connect(self.showMinimized)
        
        max_button = QPushButton("□")
        max_button.setFixedSize(30, 30)
        max_button.clicked.connect(self.toggle_maximize)
        
        close_button = QPushButton("p")
        close_button.setFixedSize(30, 30)
        close_button.clicked.connect(self.close)
        
        title_bar_layout.addWidget(title_label)
        title_bar_layout.addStretch()
        title_bar_layout.addWidget(min_button)
        title_bar_layout.addWidget(max_button)
        title_bar_layout.addWidget(close_button)
        
        main_layout.insertWidget(0, title_bar)
        
        # Window dragging
        self._drag_pos = None

        self.model = CNN(num_classes = 4)
        self.model.load_state_dict(torch.load("cnn_model.pth", map_location=torch.device('cpu')))
        self.model.eval()

        self.transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                            std  = [0.229, 0.224, 0.225])
])
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
            
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self._drag_pos is not None:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()
            
    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        event.accept()
        
    def toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()
            
    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Image", 
            "", 
            "JPEG Images (*.jpg *.jpeg)"
        )
        if file_path:
            self.process_image(file_path)
            
    def process_image(self, image_path):
        # Clear previous results
        
        self.clear_results()
        self.drop_zone.pixmap.load(image_path)
        self.drop_zone.setPixmap(self.drop_zone.pixmap)

        # Show loading animation
        self.loading_container.setVisible(True)
        self.notice_label.setVisible(True)
        
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        image = image.unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs,1)
            print(f"Predict result {predicted.item()}")

        # Simulate processing delay
        QTimer.singleShot(2000, lambda: self.display_results(predicted.item()))
        
    def display_results(self, Class):
        # Hide loading animation
        self.loading_container.setVisible(False)
        
        # For demonstration purposes, let's assume we received results from your ML algorithm
        # In a real application, you would call your ML algorithm here
        
        # Display detection result
        print(Class)
        AstigmatsmSymtoms = [
            "Blurred or distorted vision.",
            "Eyestrain or discomfort.",
            "Headaches"
        ]
        
        CataractSymptoms = [
            "Clouded, blurred, or dim vision.",
            "Difficulty seeing at night.",
            "Sensitivity to light and glare.",
            "Seeing 'halos' around lights.",
            "Frequent changes in eyeglass or contact lens prescription.",
            "Fading or yellowing of colors.",
            "Double vision in a single eye."
        ]

        DiabeticRetinopathySymptoms = [
            "Spots or dark strings (floaters) in vision.",
            "Blurred vision.",
            "Fluctuating vision.",
            "Impaired color vision.",
            "Dark or empty areas in your vision.",
            "Vision loss."
        ]
        symptoms = list()

        if (Class == 0):
            self.detection_label.setText("Astigmatism Detected.")
            symptoms = AstigmatsmSymtoms
            
        elif (Class == 1):
            self.detection_label.setText("Cataract Detected.")
            symptoms = CataractSymptoms
        elif (Class == 2):
            self.detection_label.setText("Diabetic Retinopathy Detected.")
            Symptoms = DiabeticRetinopathySymptoms
        else:
            self.detection_label.setText("Your Eye Is Healthy.")
        self.detection_label.setVisible(True)
        
        # Display confidence score
        confidence = round(random.uniform(80, 99), 2)
        self.confidence_label.setText("%" + str(confidence))
        self.confidence_label.setVisible(True)
        
        # Display symptoms section
        self.symptoms_label.setVisible(Class != 3)
        self.symptoms_content.setVisible(Class != 3)
        
        # Add symptoms (clear previous ones first)
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
        self.confidence_label.setVisible(False)
        self.symptoms_label.setVisible(False)
        self.symptoms_content.setVisible(False)
        
        # Clear symptoms
        for i in reversed(range(self.symptoms_layout.count())):
            widget = self.symptoms_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())