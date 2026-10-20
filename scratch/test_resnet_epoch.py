import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import ResNet18


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

crop_tf = T.RandomCrop(32, padding=4, padding_mode="reflect")
flip_tf = T.RandomHorizontalFlip(p=0.5)

model = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# 36000 synthetic images
imgs = torch.randn(36000, 3, 32, 32)
targets = torch.randint(0, 10, (36000,))

dataset = TensorDataset(imgs, targets)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

t0 = time.time()
model.train()
for batch_imgs, batch_targets in loader:
    batch_imgs = flip_tf(crop_tf(batch_imgs)).to(device)
    batch_targets = batch_targets.to(device)
    optimizer.zero_grad()
    outputs = model(batch_imgs)
    loss = criterion(outputs, batch_targets)
    loss.backward()
    optimizer.step()

print(f"1 Full ResNet18 Epoch Time on {device}: {time.time() - t0:.4f}s")
