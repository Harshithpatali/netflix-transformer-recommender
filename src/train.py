import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from src.models.transformer import TransformerModel

# =====================================================
# CONFIG
# =====================================================

BATCH_SIZE = 128
EPOCHS = 4
LR = 1e-3
EMBED_DIM = 32
HEADS = 2
PATIENCE = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")

# =====================================================
# LOAD DATA
# =====================================================

print("Loading training data...")

train_X = np.load("data/train_X.npy")
train_y = np.load("data/train_y.npy")

val_X = np.load("data/val_X.npy")
val_y = np.load("data/val_y.npy")

print("Train Shape:", train_X.shape)
print("Val Shape:", val_X.shape)

# =====================================================
# DATALOADER
# =====================================================

train_dataset = TensorDataset(
    torch.LongTensor(train_X),
    torch.LongTensor(train_y)
)

val_dataset = TensorDataset(
    torch.LongTensor(val_X),
    torch.LongTensor(val_y)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# =====================================================
# MODEL
# =====================================================

num_items = int(
    max(train_X.max(), val_X.max())
) + 1

print(f"Number of items: {num_items}")

model = TransformerModel(
    num_items=num_items,
    embed_dim=EMBED_DIM,
    heads=HEADS
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

best_val_loss = float("inf")
patience_counter = 0

os.makedirs("saved_models", exist_ok=True)

# =====================================================
# TRAIN LOOP
# =====================================================

for epoch in range(EPOCHS):

    # =================================================
    # TRAIN
    # =================================================

    model.train()

    train_loss = 0

    train_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1} Training"
    )

    for xb, yb in train_bar:

        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()

        logits, _ = model(xb)

        loss = criterion(logits, yb)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        train_loss += loss.item()

        train_bar.set_postfix(
            loss=loss.item()
        )

    avg_train_loss = train_loss / len(train_loader)

    # =================================================
    # VALIDATION
    # =================================================

    model.eval()

    val_loss = 0

    with torch.no_grad():

        val_bar = tqdm(
            val_loader,
            desc=f"Epoch {epoch+1} Validation"
        )

        for xb, yb in val_bar:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            logits, _ = model(xb)

            loss = criterion(logits, yb)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print("\n====================================")
    print(f"Epoch {epoch+1}")
    print(f"Train Loss : {avg_train_loss:.4f}")
    print(f"Val Loss   : {avg_val_loss:.4f}")
    print("====================================\n")

    # =================================================
    # SAVE BEST MODEL
    # =====================================================

    if avg_val_loss < best_val_loss:

        best_val_loss = avg_val_loss

        torch.save(
            model.state_dict(),
            "saved_models/transformer.pth"
        )

        print("✅ Best model saved")

    else:

        patience_counter += 1

        print(
            f"No improvement "
            f"({patience_counter}/{PATIENCE})"
        )

    # =================================================
    # EARLY STOPPING
    # =====================================================

    if patience_counter >= PATIENCE:

        print("🛑 Early stopping")
        break

print("✅ Training completed")