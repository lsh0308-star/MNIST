import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def get_args():
    parser = argparse.ArgumentParser(description="Train ResNet on MNIST")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training and testing")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--learning_rate', default=0.001, type=float, help="Learning rate for optimizer")
    parser.add_argument('--resnet_layer', type=int, default=18, help="ResNet layer options: 18, 34, 50, 101, 152")
    parser.add_argument('--pretrained', type=int, default=True, help="pretrained resnet")
    return parser.parse_args()

# Resnet layer
def get_resnet_model(layer, num_classes, pretrained):
    if layer == 18:
        model = models.resnet18(pretrained)
    elif layer == 34:
        model = models.resnet34(pretrained)
    elif layer == 50:
        model = models.resnet50(pretrained)
    elif layer == 101:
        model = models.resnet101(pretrained)
    elif layer == 152:
        model = models.resnet152(pretrained)
    else:
        raise ValueError("Invalid ResNet layer. Choose from 18, 34, 50, 101, or 152.")
    
    # Input channel 1
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # 마지막 레이어 수정
    return model

# Dataset
def get_data_loaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Train
def train(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)

# Test
def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Main
def main():
    args = get_args()
    
    # Setting
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    resnet_layer = args.resnet_layer
    pretrained = args.pretrained 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    train_loader, test_loader = get_data_loaders(batch_size)
    
    # ResNet 
    model = get_resnet_model(resnet_layer, num_classes=10, pretrained=pretrained).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train and test
    for epoch in range(epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device, epoch)
        test_accuracy = test(model, test_loader, device)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
    
    # Save
    torch.save(model.state_dict(), f'resnet{resnet_layer}_mnist.pth')
    print(f"Model saved as resnet{resnet_layer}_mnist.pth")

if __name__ == "__main__":
    main()
