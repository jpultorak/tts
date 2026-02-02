import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import tts.config as config
from tts.dataset import TTSDataset, collate_fn

# Import your custom modules
# Ensure you have installed the package via 'pip install -e .'
from tts.model import BabyValle

# --- Hyperparameters ---
BATCH_SIZE = 4  # Start small for Mac test. Increase to 16/32 on 4070 Ti
LEARNING_RATE = 3e-4  # Standard Karpathy/GPT constant
EPOCHS = 200  # For testing. Real training needs ~50-100
GRAD_CLIP = 1.0  # Prevents exploding gradients
SAVE_EVERY = 50  # Save checkpoint every N epochs
ROOT_DIR = config.ROOT_DIR
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )
    print(f"Saved checkpoint to {path}")


def train(train_data_dir):
    # 1. Setup Device
    device = get_device()
    print(f"--> Training on device: {device}")

    # Create checkpoint dir
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 2. Prepare Data
    print("--> Loading Dataset...")
    # Update this path to where your .pt files are
    train_dataset = TTSDataset(data_dir=train_data_dir)

    # On Mac/MPS, num_workers=0 is safest to avoid multiprocessing crashes
    # On Windows, you can set num_workers=4
    num_workers = 0 if device.type == "mps" else 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    print(f"--> Data loaded: {len(train_dataset)} samples.")

    # 3. Initialize Model
    model = BabyValle(
        vocab_size=2048,  # Power of 2 safety
        d_model=512,
        nhead=8,
        num_layers=6,
        max_len=4096,  # Safe margin
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 4. Training Loop
    model.train()

    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch + 1}/{EPOCHS} ===")
        total_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")

        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            logits = model(x)

            B, T, C = logits.shape
            loss = criterion(logits.view(B * T, C), y.view(B * T))

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Complete. Average Loss: {avg_loss:.4f}")

        # Save Checkpoint
        if (epoch + 1) % SAVE_EVERY == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch + 1}.pt")
            save_checkpoint(model, optimizer, epoch, avg_loss, ckpt_path)


if __name__ == "__main__":
    train(ROOT_DIR / "test_training_data")
