import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from config import Config
from data_loader import VideoDataset
from model import CNN_LSTM
from transforms import get_transforms

# Ensure multiprocessing start method is spawn for compatibility
mp.set_start_method('spawn', force=True)

# Label smoothing loss for regularization
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_probs = nn.functional.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        return ((1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss).mean()

# Set up output directory and TensorBoard logging
os.makedirs(Config.results_dir, exist_ok=True)
writer = SummaryWriter(log_dir='runs/har_fine_tune')

# Load dataset with transforms
transform = get_transforms(Config.img_size)
dataset = VideoDataset(
    Config.dataset_path,
    Config.label_map_path,
    Config.cache_path,
    transform,
    Config.frames_per_clip,
    Config.img_size
)

# Split dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

# Model setup
model = CNN_LSTM(
    num_classes=Config.num_classes,
    sequence_length=Config.frames_per_clip,
    hidden_dim=Config.hidden_dim,
    lstm_layers=Config.lstm_layers,
    dropout=Config.dropout
).to(Config.device)

# Optimizer, scheduler, and scaler
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate * 0.1, weight_decay=1e-5)  # Fine-tune with lower LR
scaler = GradScaler()

# Check if checkpoint exists
checkpoint = None
if os.path.exists(Config.model_save_path):
    checkpoint = torch.load(Config.model_save_path, map_location=Config.device)

# Automatically resume from checkpoint if exists or start from scratch
start_epoch = 0
if checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # resume from the next epoch
    print(f"Resuming fine-tuning from epoch {start_epoch}")
else:
    print("No checkpoint found. Starting fine-tuning from scratch.")

# Scheduler for fine-tuning
fine_tune_epochs = 5
scheduler = OneCycleLR(
    optimizer,
    max_lr=Config.learning_rate * 0.1,
    steps_per_epoch=len(train_loader),
    epochs=fine_tune_epochs
)

best_f1 = 0
patience = 2
counter = 0

def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)
    print(f"Checkpoint saved at {path}")

def evaluate():
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(Config.device), y.to(Config.device)
            with autocast(device_type=Config.device.type):
                out = model(x)
                loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(out.argmax(1).cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=Config.class_names, yticklabels=Config.class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.results_dir, "fine_tune_confusion_matrix.png"))
    plt.close()

    return total_loss / len(test_loader.dataset), acc, f1

def train():
    global best_f1, counter
    losses, accs = [], []

    for epoch in range(start_epoch, fine_tune_epochs):
        model.train()
        running_loss = 0

        for x, y in tqdm(train_loader, desc=f"Fine-tune Epoch {epoch+1}/{fine_tune_epochs}"):
            x, y = x.to(Config.device), y.to(Config.device)
            optimizer.zero_grad()
            with autocast(device_type=str(Config.device).split(":")[0]):
                out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        val_loss, acc, f1 = evaluate()
        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)
        accs.append(acc)

        print(f"[Fine-tune Epoch {epoch+1}/{fine_tune_epochs}] Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/val', acc, epoch)
        writer.add_scalar('F1/val', f1, epoch)

        if f1 > best_f1:
            best_f1 = f1
            counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss, Config.model_save_path)
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered!")
            break

        scheduler.step()

    save_checkpoint(model, optimizer, fine_tune_epochs, val_loss, Config.checkpoint_path)

    # Plot training metrics
    plt.plot(losses, label='Loss')
    plt.plot(accs, label='Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Fine-tuning Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.results_dir, "fine_tune_metrics.png"))
    plt.close()

    with open(os.path.join(Config.results_dir, "fine_tune_metrics.json"), "w") as f:
        json.dump({"loss": losses, "accuracy": accs}, f, indent=4)

    writer.close()

if __name__ == '__main__':
    train()
