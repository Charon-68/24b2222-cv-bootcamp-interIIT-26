import time
import torch
import torch.nn as nn

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

def batch_augment(imgs_batch):
    # imgs_batch: (B, 3, 32, 32)
    B = imgs_batch.size(0)
    # Reflection pad
    padded = nn.functional.pad(imgs_batch, (4, 4, 4, 4), mode='reflect')
    # Random crop
    tops = torch.randint(0, 9, (B,))
    lefts = torch.randint(0, 9, (B,))
    cropped = torch.stack([padded[i, :, tops[i]:tops[i]+32, lefts[i]:lefts[i]+32] for i in range(B)])
    # Random horizontal flip
    flips = torch.rand(B) > 0.5
    cropped[flips] = torch.flip(cropped[flips], dims=[3])
    return cropped

batch = torch.randn(128, 3, 32, 32, device=device)

# Warmup
_ = batch_augment(batch)

t0 = time.time()
for _ in range(281): # 281 batches = 36,000 images (1 full epoch)
    _ = batch_augment(batch)

print(f"Full Epoch Batch Augmentation Time: {time.time() - t0:.4f}s")
