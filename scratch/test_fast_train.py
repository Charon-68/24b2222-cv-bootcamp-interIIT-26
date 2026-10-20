import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import ResNet18

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Testing Fast NumPy Augment + ResNet18 on:", device)

mean_t = torch.tensor((0.4914, 0.4822, 0.4465), dtype=torch.float32).view(1, 3, 1, 1)
std_t = torch.tensor((0.2023, 0.1994, 0.2010), dtype=torch.float32).view(1, 3, 1, 1)

# Raw 36,000 uint8 images (36000, 32, 32, 3)
raw_images = np.random.randint(0, 256, (36000, 32, 32, 3), dtype=np.uint8)
targets = np.random.randint(0, 10, (36000,), dtype=np.int64)

model = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

batch_size = 128
n_batches = int(np.ceil(36000 / batch_size))

t0 = time.time()
model.train()

for b in range(n_batches):
    batch_raw = raw_images[b*batch_size : (b+1)*batch_size]
    batch_y = torch.from_numpy(targets[b*batch_size : (b+1)*batch_size]).to(device)

    B = len(batch_raw)
    padded = np.pad(batch_raw, ((0,0), (4,4), (4,4), (0,0)), mode='reflect')
    tops = np.random.randint(0, 9, size=B)
    lefts = np.random.randint(0, 9, size=B)
    cropped = np.zeros((B, 32, 32, 3), dtype=np.uint8)
    for i in range(B):
        cropped[i] = padded[i, tops[i]:tops[i]+32, lefts[i]:lefts[i]+32]
    
    flips = np.random.rand(B) > 0.5
    if np.any(flips):
        cropped[flips] = np.ascontiguousarray(cropped[flips, :, ::-1, :])
    
    t_img = torch.from_numpy(cropped).permute(0, 3, 1, 2).float() / 255.0
    batch_x = ((t_img - mean_t) / std_t).contiguous().to(device)


    optimizer.zero_grad()
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)
    loss.backward()
    optimizer.step()

print(f"1 FULL EPOCH (36,000 IMAGES) COMPLETED IN: {time.time() - t0:.2f} seconds!")
