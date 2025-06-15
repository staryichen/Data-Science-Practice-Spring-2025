# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.transforms as transforms
from dataset import StoneDataset
from models.LeNet import LeNet
from models.mlp import MLP
# from models.ResNet import resnet34
from models.AttResnet import resnet34
from PIL import Image
from sklearn.metrics import f1_score
from tools.plot import plot_cifar_distribution, plot_metrics, show_images
from torch.hub import load_state_dict_from_url
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import *
from torchvision.utils import make_grid
from tqdm import tqdm

# +
import torch.nn as nn
import torchvision.models as models

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss





# 初始化模型
model = resnet34(pretrained=True, num_classes=3, use_cbam=True)

# # 检查权重加载情况
# print("="*50)
# print("权重加载情况报告：")
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(f"{name: <40} {str(param.shape): <20} 可训练")
#     else:
#         print(f"{name: <40} {str(param.shape): <20} 冻结")
# print("="*50)

# +
num_classes = 3 

# 修改最后一层全连接层
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 冻结所有层
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# +
# root = 'dataset'
batch_size = 128 ##256--

# # 定义数据集的预处理步骤


transforms_train = transforms.Compose(
    [
        transforms.RandomRotation(60),
        transforms.Resize((224, 224)),#resize256cutinto224?
        transforms.CenterCrop((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),#del?
        transforms.RandomAffine(degrees=0, shear=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), value="random"),
    ]
)

transforms_test = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Create datasets
train_dataset = StoneDataset(
    "dataset/train_val/", split="train", transforms=transforms_train
)
val_dataset = StoneDataset(
    "dataset/train_val/", split="val", transforms=transforms_test
)
test_dataset = StoneDataset("dataset/test/", split="test", transforms=transforms_test)

# Create data loaders
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)

# Check dataset
print(f"Train batches: {len(train_loader)}")
print(f"Valid batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")
# print(f"Number of classes: {len(train_dataset.class_to_idx)}")

# +
# 查看类别分布
def analyze_class_distribution(dataset):
    if dataset.split == "test":
        print("测试集无标签信息")
        return

    labels = dataset.labels
    class_counts = {0: 0, 1: 0, 2: 0}
    for label in labels:
        class_counts[label] += 1

    print(f"{dataset.split}集类别分布:")
    for cls, count in class_counts.items():
        print(f"类别 {cls}: {count} 张图片 ({count/len(dataset)*100:.2f}%)")


analyze_class_distribution(train_dataset)
analyze_class_distribution(val_dataset)

# +
# def compute_image_stats(dataloader):
#     mean = 0.
#     std = 0.
#     nb_samples = 0.

#     for data, _ in tqdm(dataloader):
#         batch_samples = data.size(0)
#         data = data.view(batch_samples, data.size(1), -1)
#         mean += data.mean(2).sum(0)
#         std += data.std(2).sum(0)
#         nb_samples += batch_samples

#     mean /= nb_samples
#     std /= nb_samples

#     return mean, std

# # 计算训练集的均值和标准差
# train_mean, train_std = compute_image_stats(train_loader)
# print(f"训练集均值: {train_mean}, 标准差: {train_std}")

"""
100%|██████████| 200/200 [10:21<00:00,  3.11s/it]
训练集均值: tensor([-0.2765, -0.1739,  0.0217]), 标准差: tensor([0.9080, 0.9321, 0.9443])
"""


# 可视化样本图像
def show_sample_images(dataloader, num_images=8):
    images, labels = next(iter(dataloader))
    grid = make_grid(images[:num_images], nrow=4, normalize=True)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title("Sample Training Images")
    plt.axis("off")
    plt.show()


#show_sample_images(train_loader)


# +
# 测试函数
def evaluate(model, data_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            # 如果是测试集（labels是路径），跳过评估
            if isinstance(labels[0], str):
                return (
                    0,
                    0,
                    0,
                )  # 或者 raise ValueError("Cannot evaluate on test set without labels")

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = test_loss / len(data_loader)
    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, accuracy, f1


# def evaluate(model, test_loader, criterion, device):
#     model.eval()
#     total_loss = 0
#     total_correct = 0
#     total_samples = 0
#     preds = []
#     label = []

#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             total_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total_samples += labels.size(0)
#             total_correct += (predicted == labels).sum().item()

#             preds.extend(predicted.cpu().numpy())
#             label.extend(labels.cpu().numpy())

#     avg_loss = total_loss / len(test_loader)
#     accuracy = total_correct / total_samples
#     macro_f1 = f1_score(label, preds, average='macro')
#     return avg_loss, accuracy, macro_f1

# +
# 超参数
num_epochs = 40
learning_rate = 0.001##delete

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# +
model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

num_classes = 3
model.fc = nn.Linear(model.fc.in_features, num_classes)

params_group = [
    {"params": [p for n, p in model.named_parameters() if "cbam" not in n and "fc" not in n], "lr": 0.005},
    {"params": [p for n, p in model.named_parameters() if "cbam" in n], "lr": 0.008},  # 注意力层更高学习率
    {"params": model.fc.parameters(), "lr": 0.01},
]

criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
optimizer = optim.SGD(params_group, momentum=0.9, weight_decay=1e-4, nesterov=True)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# -

# 训练一个batch的函数
def train_one_batch(model, inputs, labels, optimizer, criterion, device):
    """
    Args:
        model (nn.Module): 神经网络模型
        inputs (torch.Tensor): 输入数据，形状为 (batch_size, channels, height, width)
        labels (torch.Tensor): 真实标签，形状为 (batch_size,)
        optimizer (torch.optim.Optimizer): 优化器（如 torch.optim.SGD）
        criterion (nn.Module): 损失函数（如 nn.CrossEntropyLoss）
        device (torch.device): 运行设备（'cpu' 或 'cuda'）

    Returns:
        tuple: (loss, outputs)
            - loss (torch.Tensor): 当前批次的损失值（标量）
            - outputs (torch.Tensor): 模型的输出，形状为 (batch_size, num_classes)
    """
    # 1. 将inputs和labels移动到指定设备device
    inputs, labels = inputs.to(device), labels.to(device)

    # 2. 重置优化器optimizer的梯度
    optimizer.zero_grad()

    # 3. 前向传播：通过模型model计算输出
    outputs = model(inputs)

    # 4. 计算损失loss
    loss = criterion(outputs, labels)

    # 5. 反向传播：计算梯度
    loss.backward()

    # 6. 更新参数
    optimizer.step()

    return loss, outputs


# +
# 记录训练过程中的指标
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
val_f1scores = []
learning_rates = []
early_stopping = EarlyStopping(patience=5, verbose=True)

# 训练循环
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    epoch_correct = 0
    total_samples = 0

    # 按batch训练
    for i, batch_data in enumerate(train_loader):
        inputs, labels = batch_data
        inputs, labels = inputs.to(device), labels.to(device)
        batch_loss, batch_output = train_one_batch(
            model, inputs, labels, optimizer, criterion, device
        )
        epoch_loss += batch_loss.item()

        # 1. 从 batch_output 中获取预测类别 (使用 torch.max)
        # 2. 获取当前批次大小 (labels 的第 0 维大小)
        # 3. 更新 total_samples (累加批次大小)
        # 4. 计算正确预测的数量并更新 epoch_correct

        _, batch_prediction = torch.max(batch_output.data, 1)
        batch_size = labels.size(0)
        total_samples += batch_size
        epoch_correct += (batch_prediction == labels).sum().item()

    # 计算epoch loss
    avg_train_loss = epoch_loss / len(train_loader)

    # 使用 epoch_correct 和 total_samples 计算平均准确率
    avg_train_acc = epoch_correct / total_samples

    # 测试
    val_loss, val_acc, f1score = evaluate(model, val_loader, criterion, device)

    # 记录指标
    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    learning_rates.append(scheduler.get_last_lr()[0])  # 记录当前学习率
    val_f1scores.append(f1score)
    # 每个 epoch 结束后使用scheduler更新学习率
    scheduler.step()
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break
    
    # 打印训练状态
    if (epoch + 1) % 1 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
        print(
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Macro F1: {f1score:.4f}"
        )
    print(f"Current Learning Rate: {scheduler.get_last_lr()[0]:.6f}")  # 打印当前学习率
model.load_state_dict(torch.load('checkpoint.pt'))
# -

plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies, num_epochs)



# +
### How to Generate submission.csv from test_loader


#### 1. **Define the Prediction Function**


def predict(model, loader, device):
    model.eval()  # Set the model to evaluation mode
    predictions = []  # Store predicted classes
    image_ids = []  # Store image filenames

    with torch.no_grad():  # Disable gradient computation
        for images, img_paths in tqdm(loader, desc="Predicting on test set"):
            images = images.to(device)  # Move images to the specified device
            outputs = model(images)  # Forward pass to get model outputs
            _, predicted = torch.max(outputs, 1)  # Get predicted classes

            # Collect predictions and image IDs
            predictions.extend(predicted.cpu().numpy())
            image_ids.extend([os.path.basename(path) for path in img_paths])

    return image_ids, predictions


#### 2. **Run Predictions**
image_ids, predictions = predict(model, test_loader, device)

# +
#### 3. **Create the Submission File**

import os

import pandas as pd

# Create DataFrame
submission_df = pd.DataFrame(
    {"id": image_ids, "label": predictions}  # Image filenames  # Predicted classes
)

# Save to the specified path
OUTPUT_DIR = "logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
submission_path = os.path.join(OUTPUT_DIR, "new.csv")
submission_df.to_csv(submission_path, index=False)
print(f"Kaggle submission file saved to {submission_path}")
# -


