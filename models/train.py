import torch
import torch.nn as nn
import torch.optim as optim
from models.genconvit_ed import GenConViTED
from models.config import load_config
from data_loader import load_data, load_checkpoint

# Load config
config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = GenConViTED(config, pretrained=True)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
      model.parameters(),
      lr=config['learning_rate'],
      weight_decay=config['weight_decay']
)

  # Load data
dataloaders, dataset_sizes = load_data(
      data_dir='/Users/lobeli/Desktop/PROJECT/GenConViT/dataset',
      batch_size=config['batch_size']
)

  # Training loop
num_epochs = config['epoch']  # Update in config.yaml
best_loss = float('inf')

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')

    # Training phase
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloaders['train']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass (calls genconvit_ed.py:119-131)
        outputs = model(inputs)  # [batch, 2]
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_sizes['train']
    epoch_acc = running_corrects.double() / dataset_sizes['train']

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloaders['validation']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)

    val_loss = val_loss / dataset_sizes['valid']
    val_acc = val_corrects.double() / dataset_sizes['valid']

    print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

      # Save best model
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'min_loss': best_loss
        }, 'model/GenConViT/genconvit_ed_best.pth')