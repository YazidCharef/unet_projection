from data_processing import get_data_loaders
from train import train_model
from evaluate import evaluate_model
from models.unet import TemporalUNet
from config import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, DEVICE, SEQUENCE_LENGTH
import torch
import torch.nn as nn
import torch.optim as optim

def main():
    print("Step 1: Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=BATCH_SIZE, data_fraction=0.1)

    print("Step 2: Initializing model...")
    model = TemporalUNet(n_channels=2, n_classes=1, n_frames=SEQUENCE_LENGTH).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Step 3: Training model...")
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, DEVICE)

    print("Step 4: Saving trained model...")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'temporal_typhoon_unet_model_final.pth')

    print("Step 5: Evaluating model...")
    test_loss, predictions, targets = evaluate_model(trained_model, test_loader, criterion, DEVICE)

    print(f"Test Loss: {test_loss:.4f}")
    print("Visualization of predictions saved.")

    print("All steps completed successfully.")

if __name__ == "__main__":
    main()