import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# 定义数据集类
class AnimalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        # 遍历每个动物文件夹
        for animal in os.listdir(root_dir):
            animal_dir = os.path.join(root_dir, animal)
            for img_name in os.listdir(animal_dir):
                img_path = os.path.join(animal_dir, img_name)
                # 根据动物名称设置标签，猫为1，狗为2，其他为0
                label = 1 if animal == "cat" else 2 if animal == "dog" else 0
                self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")  # 确保图像是RGB格式
        if self.transform:
            image = self.transform(image)
        return image, label


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


# 数据增强和归一化
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# 创建数据集
dataset = AnimalDataset(root_dir="animal-10", transform=transform)

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size]
)

# 创建数据加载器，设置num_workers以利用多线程数据加载
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)

# 确定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AnimalClassifier(num_classes=3).to(device)

# 创建损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 控制进度条显示的参数
show_progress = True

# 训练循环
num_epochs = 10  # 设置训练的轮数
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    if show_progress:
        train_loader_tqdm = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
        )
    else:
        train_loader_tqdm = train_loader

    for images, labels in train_loader_tqdm:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if show_progress:
            train_loader_tqdm.set_postfix(loss=running_loss / len(train_loader))

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

    # 验证过程
    model.eval()
    correct = 0
    total = 0
    if show_progress:
        test_loader_tqdm = tqdm(test_loader, desc="Validation", leave=False)
    else:
        test_loader_tqdm = test_loader

    with torch.no_grad():
        for images, labels in test_loader_tqdm:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the model on the test images: {100 * correct / total}%")
