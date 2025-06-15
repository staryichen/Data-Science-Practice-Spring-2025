from PIL import Image, UnidentifiedImageError
import os
import pandas as pd
from torch.utils.data import Dataset

class StoneDataset(Dataset):
    def __init__(self, root, split="train", transforms=None):
        """
        root: 数据集根目录
        split: 'train', 'val', 或 'test'
        transforms: 图像预处理变换
        """
        # Train size: 102213, Val size: 15000, Test size: 15000
        self.root = root
        self.split = split
        self.transforms = transforms
        self.samples = []
        self.labels = []

        # 根据 split 加载对应的数据
        if split in ["train", "val"]:
            # 加载训练集或验证集
            csv_path = os.path.join(root, f"{split}_labels.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"{csv_path} not found.")
            
            # 读取 CSV 文件
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                img_path = os.path.join(root, split, row["id"])
                self.samples.append(img_path)
                self.labels.append(int(row["label"]))  # label 已为 0, 1, 2
        elif split == "test":
            # 加载测试集（无标签，仅图像路径）
            test_ids_path = os.path.join(root, "test_ids.csv")
            if not os.path.exists(test_ids_path):
                raise FileNotFoundError(f"{test_ids_path} not found.")
            
            # 读取测试集 ID
            df = pd.read_csv(test_ids_path)
            for _, row in df.iterrows():
                img_path = os.path.join(root, "test_images", row["id"])
                self.samples.append(img_path)
                self.labels.append(None)  # 测试集无标签，占位符
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    def __getitem__(self, index):
        img_path = self.samples[index]
        label = self.labels[index]

        # 加载图像
        image = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        # 对于测试集，label 为 None，仅返回图像
        if self.split == "test":
            return image, img_path  # 返回图像和路径以便生成提交文件
        return image, label  # 返回图像和标签

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":

    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载数据集
    dataset_train = StoneDataset(root="./dataset/train_val", split="train", transforms=transform)
    dataset_val = StoneDataset(root="./dataset/train_val", split="val", transforms=transform)
    dataset_test = StoneDataset(root="./dataset/test", split="test", transforms=transform)   # for Kaggle test only

    print(f"Train size: {len(dataset_train)}")
    print(f"Val size: {len(dataset_val)}")
    print(f"Test size: {len(dataset_test)}")

    # 测试加载
    img, label = dataset_train[0]
    print(f"Sample image shape: {img.shape}, Label: {label}")