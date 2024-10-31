import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models.unet import TemporalUNet
from data_processing import get_data_loaders
from config import DEVICE, BATCH_SIZE

def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)

            predictions = torch.sigmoid(outputs)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')

    return all_predictions, all_targets

def visualize_predictions(predictions, targets, index):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for i in range(9):
        ax = axes[i // 3, i % 3]
        ax.imshow(predictions[index, 0, i], cmap='binary')
        ax.set_title(f'Frame {i+1}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'prediction_{index}.png')
    plt.close()

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for i in range(9):
        ax = axes[i // 3, i % 3]
        ax.imshow(targets[index, i], cmap='binary')
        ax.set_title(f'Frame {i+1}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'target_{index}.png')
    plt.close()

