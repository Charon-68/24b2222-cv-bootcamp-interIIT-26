import time
import torch
import torchvision.transforms as T

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

crop_tf = T.RandomCrop(32, padding=4, padding_mode="reflect")
flip_tf = T.RandomHorizontalFlip(p=0.5)

# CPU batch input
batch_cpu = torch.randn(128, 3, 32, 32)

t0 = time.time()
for _ in range(281): # 1 full epoch (36,000 images)
    augmented_cpu = flip_tf(crop_tf(batch_cpu))
    batch_gpu = augmented_cpu.to(device, non_blocking=True)

print(f"CPU Augment + GPU Transfer per Epoch Time: {time.time() - t0:.4f}s")
