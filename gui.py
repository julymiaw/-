import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import numpy as np


# 定义模型类
class AnimalClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(AnimalClassifier, self).__init__()
        # 使用预训练的ResNet18，并修改最后的全连接层
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x


# 加载最优模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AnimalClassifier(num_classes=3).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# 定义图像预处理
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# 定义类别标签
class_names = ["其他", "猫", "狗"]


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("动物分类器")

        # 设置布局
        layout = QVBoxLayout()

        # 创建标签
        self.label = QLabel("请选择一张图片进行分类", self)
        self.label.setFont(QFont("Arial", 12))
        layout.addWidget(self.label)

        # 创建显示图片的标签
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(300, 300)
        layout.addWidget(self.image_label)

        # 创建按钮
        self.button = QPushButton("选择图片", self)
        self.button.setFont(QFont("Arial", 12))
        self.button.clicked.connect(self.load_image)
        layout.addWidget(self.button)

        # 创建结果标签
        self.result_label = QLabel("", self)
        self.result_label.setFont(QFont("Arial", 12))
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "Images (*.png *.xpm *.jpg *.bmp *.gif *.jpeg)",
            options=options,
        )
        if file_path:
            image = Image.open(file_path).convert("RGB")
            self.display_image(image)
            self.predict(image)

    def display_image(self, image):
        image = image.resize((300, 300), Image.LANCZOS)
        image_np = np.array(image)
        height, width, channel = image_np.shape
        bytes_per_line = 3 * width
        q_image = QImage(
            image_np.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

    def predict(self, image):
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            class_name = class_names[predicted.item()]
            self.result_label.setText(f"预测结果: {class_name}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
