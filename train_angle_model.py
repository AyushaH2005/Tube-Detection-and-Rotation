import os
import math
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------
# Dataset
# --------------------------------

class AngleDataset(Dataset):

    def __init__(self, csv_file, split):

        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df["split"] == split]

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img_path = os.path.join(
            "crops",
            row["split"],
            row["file"]
        )

        image = Image.open(img_path).convert("RGB")

        image = self.transform(image)

        angle_rad = math.radians(row["angle"])

        target = torch.tensor([
            math.cos(angle_rad),
            math.sin(angle_rad)
        ], dtype=torch.float32)

        return image, target

# --------------------------------
# DataLoaders
# --------------------------------

train_dataset = AngleDataset(
    "angle_labels.csv",
    "train"
)

val_dataset = AngleDataset(
    "angle_labels.csv",
    "val"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16
)

# --------------------------------
# Model
# --------------------------------

model = models.resnet18(weights="DEFAULT")

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model = model.to(DEVICE)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

# --------------------------------
# Training
# --------------------------------

EPOCHS = 10

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    for images, targets in train_loader:

        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {avg_loss:.4f}")

# --------------------------------
# Save model
# --------------------------------

os.makedirs("models", exist_ok=True)

torch.save(
    model.state_dict(),
    "models/angle_model.pth"
)

print("Angle model saved successfully!")