import time
import torch
import numpy as np

# Batch of 128 uint8 images (128, 32, 32, 3)
imgs_np = np.random.randint(0, 256, (128, 32, 32, 3), dtype=np.uint8)

mean = torch.tensor((0.4914, 0.4822, 0.4465), dtype=torch.float32).view(1, 3, 1, 1)
std = torch.tensor((0.2023, 0.1994, 0.2010), dtype=torch.float32).view(1, 3, 1, 1)

def fast_batch_np(imgs):
    B = len(imgs)
    padded = np.pad(imgs, ((0,0), (4,4), (4,4), (0,0)), mode='reflect')
    tops = np.random.randint(0, 9, size=B)
    lefts = np.random.randint(0, 9, size=B)
    cropped = np.zeros((B, 32, 32, 3), dtype=np.uint8)
    for i in range(B):
        cropped[i] = padded[i, tops[i]:tops[i]+32, lefts[i]:lefts[i]+32]
    
    # Flips
    flips = np.random.rand(B) > 0.5
    cropped[flips] = cropped[flips, :, ::-1, :]
    
    # Convert to normalized float tensor
    t = torch.from_numpy(cropped).permute(0, 3, 1, 2).float() / 255.0
    return (t - mean) / std

t0 = time.time()
for _ in range(281):
    _ = fast_batch_np(imgs_np)

print(f"281 batches fast numpy augment time: {time.time() - t0:.4f}s")
