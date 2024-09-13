# 动物分类器项目

本项目包含两个主要的Python脚本：`train.py` 和 `gui.py`。`train.py` 用于训练一个动物分类模型，而 `gui.py` 提供了一个图形用户界面，用于加载图像并使用训练好的模型进行分类。

## 目录结构

```plaintext
.
├── animal-10
│   ├── butterfly
│   ├── cat
│   ├── chicken
│   ├── cow
│   ├── dog
│   ├── elephant
│   ├── horse
│   ├── sheep
│   ├── spider
│   └── squirrel
├── best_model.pth
├── gui.py
├── README.md
└── train.py
```

## `train.py`

`train.py` 脚本用于训练一个基于ResNet18的动物分类模型。主要步骤如下：

1. **导入必要的库**：

    ```python
    import os
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models
    from PIL import Image
    import torch.nn as nn
    import torch.optim as optim
    from tqdm import tqdm
    ```

2. **定义数据集类**：

    ```python
    class AnimalDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            # 初始化代码
        def __len__(self):
            # 返回数据集大小
        def __getitem__(self, idx):
            # 获取数据和标签
    ```

3. **定义模型类**：

    ```python
    class AnimalClassifier(nn.Module):
        def __init__(self, num_classes=3):
            # 初始化模型
        def forward(self, x):
            # 前向传播
    ```

4. **数据增强和归一化**：

    ```python
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ```

5. **创建数据集和数据加载器**：

    ```python
    dataset = AnimalDataset(root_dir="animal-10", transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)
    ```

6. **训练模型**：

    ```python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnimalClassifier(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    num_epochs = 10
    for epoch in range(num_epochs):
        # 训练循环
        # 验证过程
        # 保存最佳模型
    ```

## [`gui.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fjuly%2F%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%2Farchive%2Fgui.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22d38cb190-a7aa-4b1b-8526-bd33c05d4c1f%22%5D "/home/july/计算机视觉/archive/gui.py")

[`gui.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fjuly%2F%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%2Farchive%2Fgui.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22d38cb190-a7aa-4b1b-8526-bd33c05d4c1f%22%5D "/home/july/计算机视觉/archive/gui.py") 脚本提供了一个图形用户界面，用于加载图像并使用训练好的模型进行分类。主要步骤如下：

1. **导入必要的库**：

    ```python
    import sys
    from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout
    from PyQt5.QtGui import QPixmap, QImage, QFont
    from PIL import Image
    import torch
    from torchvision import transforms, models
    import torch.nn as nn
    import numpy as np
    ```

2. **定义模型类**：

    ```python
    class AnimalClassifier(nn.Module):
        def __init__(self, num_classes=3):
            # 初始化模型
        def forward(self, x):
            # 前向传播
    ```

3. **加载最优模型**：

    ```python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnimalClassifier(num_classes=3).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()
    ```

4. **定义图像预处理**：

    ```python
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ```

5. **定义类别标签**：

    ```python
    class_names = ["其他", "猫", "狗"]
    ```

6. **创建图形用户界面**：

    ```python
    class App(QWidget):
        def __init__(self):
            # 初始化界面
        def initUI(self):
            # 设置界面布局
        def load_image(self):
            # 加载图像
        def display_image(self, image):
            # 显示图像
        def predict(self, image):
            # 预测图像类别
    ```

7. **运行应用程序**：

    ```python
    if __name__ == "__main__":
        app = QApplication(sys.argv)
        ex = App()
        ex.show()
        sys.exit(app.exec_())
    ```

## 运行说明

1. **训练模型**：

    ```bash
    python train.py
    ```

2. **运行图形用户界面**：

    ```bash
    python gui.py
    ```

确保在运行 [`gui.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fjuly%2F%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%2Farchive%2Fgui.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22d38cb190-a7aa-4b1b-8526-bd33c05d4c1f%22%5D "/home/july/计算机视觉/archive/gui.py") 之前已经训练并保存了模型 `best_model.pth`。

## 依赖项

- Python 3.x
- PyTorch
- torchvision
- PIL (Pillow)
- tqdm
- PyQt5
- numpy
