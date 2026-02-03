import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

import tts.config as config
from tts.dataset import TTSDataset, collate_fn

# Import your custom modules
from tts.model import BabyValle

# --- Hyperparameters (Optimized for 4070 Ti Overnight) ---
BATCH_SIZE = 24       # Safe sweet spot. 
LEARNING_RATE = 3e-4  
EPOCHS = 100          # The goal for a good model
GRAD_CLIP = 1.0       
SAVE_EVERY = 5        # Save often enough to not lose progress
NUM_WORKERS = 6      # Sweet spot for Windows

ROOT_DIR = config.ROOT_DIR
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
DATA_DIR = ROOT_DIR / "data"
RESUME_FROM = CHECKPOINT_DIR / "model_epoch_10.pt" 

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def save_checkpoint(model, optimizer, scaler, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(), 
        'loss': loss,
    }, path)
    print(f"Saved checkpoint to {path}")

def load_checkpoint(path, model, optimizer, scaler, device):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])

    start_epoch = checkpoint['epoch']
    loss = checkpoint.get('loss', None)

    print(f"--> Loaded checkpoint from {path} (epoch {start_epoch})")
    if loss is not None:
        print(f"--> Checkpoint loss: {loss:.4f}")

    return start_epoch

def train():
    # 1. Setup Device
    device = get_device()
    print(f"--> Training on device: {device}")
    
    # 2. Setup AMP (The Speed Booster) - NEW SYNTAX
    # We only use AMP if we are strictly on CUDA for now.
    use_amp = (device.type == 'cuda')
    
    # Initialize the Scaler with the specific device
    # "enabled=use_amp" ensures it does nothing if we aren't on GPU
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    if use_amp:
        print("--> Automatic Mixed Precision (AMP) Enabled 🚀")

    # Create checkpoint dir
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 3. Prepare Data
    print(f"--> Loading Dataset from {DATA_DIR}...")
    train_dataset = TTSDataset(data_dir=DATA_DIR)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        # Pin memory speeds up transfer from RAM to GPU
        pin_memory=True if device.type == 'cuda' else False 
    )
    print(f"--> Data loaded: {len(train_dataset)} samples.")

    # 4. Initialize Model
    model = BabyValle(
        vocab_size=2048, 
        d_model=512,
        nhead=8,
        num_layers=6,
        max_len=4096     
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    start_epoch = 0
    if RESUME_FROM is not None and RESUME_FROM.exists():
        start_epoch = load_checkpoint(
            RESUME_FROM, model, optimizer, scaler, device
        )

    # 5. Training Loop
    model.train()

    for epoch in range(start_epoch, EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
        total_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")

        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            
            # --- AMP Forward Pass ---
            # NEW SYNTAX: torch.amp.autocast('cuda', ...)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(x)
                B, T, C = logits.shape
                loss = criterion(logits.view(B*T, C), y.view(B*T))

            optimizer.zero_grad()
            
            # --- AMP Backward Pass ---
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Complete. Average Loss: {avg_loss:.4f}")

        # Save Checkpoint
        if (epoch + 1) % SAVE_EVERY == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pt")
            save_checkpoint(model, optimizer, scaler, epoch + 1, avg_loss, ckpt_path)

if __name__ == "__main__":
    train()