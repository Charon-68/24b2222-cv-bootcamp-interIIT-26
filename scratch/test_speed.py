import time
import torch
import torch.nn as nn
from torchvision import transforms as T
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

# Load 1000 images
imgs = np.random.randint(0, 256, (1000, 32, 32, 3), dtype=np.uint8)
t_start = time.time()
raw_t = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0
print(f"Pre-tensor conversion: {time.time() - t_start:.4f}s")

crop = T.RandomCrop(32, padding=4, padding_mode="reflect")
flip = T.RandomHorizontalFlip(p=0.5)

t_start = time.time()
for i in range(1000):
    img = raw_t[i]
    img = crop(img)
    img = flip(img)
print(f"1000 individual transforms time: {time.time() - t_start:.4f}s")
