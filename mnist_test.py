import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from mnist_train import get_args

args = get_args()

# ResNet layer
def get_resnet_model(layer, num_classes, pretrained):
    if layer == 18:
        model = models.resnet18(pretrained=pretrained)
    elif layer == 34:
        model = models.resnet34(pretrained=pretrained)
    elif layer == 50:
        model = models.resnet50(pretrained=pretrained)
    elif layer == 101:
        model = models.resnet101(pretrained=pretrained)
    elif layer == 152:
        model = models.resnet152(pretrained=pretrained)
    else:
        raise ValueError("Invalid ResNet layer. Choose from 18, 34, 50, 101, or 152.")
    
    # Input channel 1
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # 마지막 레이어 수정
    return model

# Dataset load
def get_test_data_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True)
    return test_loader

# Test
def test(model, loader, device):
    model.eval()
    images, labels = next(iter(loader))  # 10장 가져오기
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    return images, labels, predicted

# Main
def main():
    # Setting
    model_path = 'resnet18_mnist.pth'  # Model directory
    resnet_layer = args.resnet_layer 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrained = args.pretrained

    model = get_resnet_model(resnet_layer, num_classes=10, pretrained=pretrained).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")

    test_loader = get_test_data_loader()
    images, labels, predicted = test(model, test_loader, device)

    # Visualize
    images = images.cpu()
    labels = labels.cpu()
    predicted = predicted.cpu()

    plt.figure(figsize=(12, 6))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].squeeze(0), cmap='gray')
        plt.title(f"True: {labels[i].item()} | Pred: {predicted[i].item()}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
