
#!/usr/bin/env python

import argparse
from time import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision import transforms
from medmnist import INFO
import medmnist

import src as models
from examples.utils.losses import LabelSmoothingCrossEntropy

def get_medmnist_loaders(data_flag='pneumoniamnist', batch_size=128):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    train_dataset = DataClass(split='train', transform=transform, download=True)
    val_dataset = DataClass(split='val', transform=transform, download=True)
    test_dataset = DataClass(split='test', transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, len(info['label'])

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))
    return [correct[:, :k].float().sum().mul_(100.0 / batch_size) for k in topk]

def main():
    parser = argparse.ArgumentParser(description='MedMNIST with CCT')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--model', default='cct_7', type=str)
    parser.add_argument('--dataset', default='pneumoniamnist', type=str)
    args = parser.parse_args()

    # Load data
    train_loader, val_loader, test_loader, num_classes = get_medmnist_loaders(args.dataset, args.batch_size)

    # Create model
    model = getattr(models, args.model)(pretrained=False, num_classes=num_classes, in_chans=1)
    model = model.cuda()

    # Loss and optimizer
    criterion = LabelSmoothingCrossEntropy()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)

        print(f"Epoch {epoch+1:02d}: Train Acc: {100 * correct / total:.2f}%, Loss: {total_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            pred = outputs.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == '__main__':
    main()
