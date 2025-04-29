import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

class FlowerDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = self._load_labels(label_file)

    def _load_labels(self, label_file):
        with open(label_file, "r") as f:
            lines = f.readlines()
        img_labels = [line.strip().split() for line in lines]
        return img_labels

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name, label = self.img_labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(label)
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs.half())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total
    return train_loss, train_acc


def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.half())
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = running_loss / len(test_loader)
    test_acc = 100.0 * correct / total
    return test_loss, test_acc


def main():
    train_dataset = FlowerDataset(
        img_dir=r"/home/mozihao/CLS/Oxford_Flowers_102/train",
        label_file=r"/home/mozihao/CLS/Oxford_Flowers_102/train/labels.txt",
        transform=transform
    )
    test_dataset = FlowerDataset(
        img_dir=r"/home/mozihao/CLS/Oxford_Flowers_102/test",
        label_file=r"/home/mozihao/CLS/Oxford_Flowers_102/test/labels.txt",
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
   
    num_classes = 102 
    # model = models.resnet18(pretrained=True)
    # model.fc = nn.Linear(model.fc.in_features, num_classes)

    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)

    # model = models.mobilenet_v3_large(pretrained=True)
    # in_features = model.classifier[-1].in_features
    # model.classifier[-1] = nn.Linear(in_features, num_classes)

    # model = models.efficientnet_b4(pretrained=True)
    # in_features = model.classifier[-1].in_features
    # model.classifier[-1] = nn.Linear(in_features, num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device).half()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, min_lr=1e-6)

    model_structure(model)

    # Train
    num_epochs = 100
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        with open("log/mobilenet_v2_half(resize_224+norm+flip+rotation+lr_scheduler).txt","a") as log:
            log.write(f"Epoch [{epoch + 1}/{num_epochs}], lr: {optimizer.param_groups[0]['lr']:.6f}, "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%\n")
        scheduler.step(test_loss)

    print("Finished training !")

if __name__ == "__main__":
    main()