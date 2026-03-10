# train_model.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import nibabel as nib
import numpy as np
import pandas as pd

# ================= CONFIGURATION =================
DATA_DIR = "data/OASIS1"
CSV_PATH = "data/oasis1_demographics.xlsx"   # <-- change if needed
MODEL_SAVE_PATH = "models/brain_age_model.pth"

IMAGE_SIZE = 64
BATCH_SIZE = 2
EPOCHS = 15
LEARNING_RATE = 0.0005
TRAIN_SPLIT = 0.8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ================= DATASET =================
class MRIDataset(Dataset):
    def __init__(self, data_dir, csv_path):
        self.data_dir = data_dir

        # Load CSV or Excel
        if csv_path.endswith(".xlsx"):
            self.df = pd.read_excel(csv_path)
        else:
            self.df = pd.read_csv(csv_path, sep=None, engine="python")

        self.df.columns = self.df.columns.str.strip()
        self.df = self.df.dropna(subset=["Age"])

        self.samples = []

        for _, row in self.df.iterrows():
            subject_id = str(row["ID"]).strip()
            age = float(row["Age"])

            # 🔥 Recursively search for .img files
            for root, dirs, files in os.walk(self.data_dir):
                for file in files:
                    if file.startswith(subject_id) and file.endswith(".img"):
                        file_path = os.path.join(root, file)
                        self.samples.append((file_path, age))
                        break

        if len(self.samples) == 0:
            raise ValueError(
                f"❌ No .img MRI files found inside {self.data_dir}"
            )

        print(f"✅ Total matched MRI samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, age = self.samples[idx]

        img = nib.load(file_path).get_fdata()

        # 🔥 FIX: Remove extra 4th dimension if exists
        if len(img.shape) == 4:
            img = img[:, :, :, 0]

        # Normalize safely
        mean = np.mean(img)
        std = np.std(img)
        if std == 0:
            std = 1.0
        img = (img - mean) / std

        img = torch.tensor(img, dtype=torch.float32)

        # Add batch & channel dimensions
        img = img.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)

        # Resize correctly
        img = F.interpolate(
            img,
            size=(64, 64, 64),
            mode="trilinear",
            align_corners=False
        )

        img = img.squeeze(0)  # Remove batch dim → (1,64,64,64)

        age = torch.tensor(age, dtype=torch.float32)

        return img, age
# ================= MODEL =================
class BrainAgeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )

        self.regressor = nn.Linear(32, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)


# ================= LOAD DATA =================
dataset = MRIDataset(DATA_DIR, CSV_PATH)

train_size = int(TRAIN_SPLIT * len(dataset))
val_size = len(dataset) - train_size

if val_size <= 0:
    raise ValueError("❌ Dataset too small for splitting.")

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ================= TRAIN SETUP =================
model = BrainAgeModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("🚀 Starting training...\n")


# ================= TRAIN LOOP =================
for epoch in range(EPOCHS):

    # ---- Training ----
    model.train()
    train_loss = 0.0

    for images, ages in train_loader:
        images = images.to(device)
        ages = ages.to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze()

        loss = criterion(outputs, ages)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---- Validation ----
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, ages in val_loader:
            images = images.to(device)
            ages = ages.to(device)

            outputs = model(images).squeeze()
            loss = criterion(outputs, ages)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"| Train Loss: {train_loss:.4f} "
          f"| Val Loss: {val_loss:.4f}")


# ================= SAVE MODEL =================
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)

print("\n✅ Training complete!")
print(f"📦 Model saved at: {MODEL_SAVE_PATH}")
