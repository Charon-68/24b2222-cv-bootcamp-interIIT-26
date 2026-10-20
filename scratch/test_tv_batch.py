import time
import torch
import torchvision.transforms as T

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

crop_tf = T.RandomCrop(32, padding=4, padding_mode="reflect")
flip_tf = T.RandomHorizontalFlip(p=0.5)

def batch_tf(imgs_batch):
    return flip_tf(crop_tf(imgs_batch))

batch = torch.randn(128, 3, 32, 32, device=device)

# Warmup
_ = batch_tf(batch)

t0 = time.time()
for _ in range(281):
    _ = batch_tf(batch)

print(f"Native Torchvision 4D Batch Augmentation Time: {time.time() - t0:.4f}s")
