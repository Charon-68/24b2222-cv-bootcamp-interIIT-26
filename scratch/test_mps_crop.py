import time
import torch
import torchvision.transforms as T

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

crop_tf = T.RandomCrop(32, padding=4, padding_mode="reflect")

# CPU test
b_cpu = torch.randn(128, 3, 32, 32)
t0 = time.time()
out_cpu = crop_tf(b_cpu)
print(f"CPU crop time: {time.time() - t0:.4f}s")

# MPS test
b_mps = torch.randn(128, 3, 32, 32, device=device)
t0 = time.time()
try:
    out_mps = crop_tf(b_mps)
    print(f"MPS crop time: {time.time() - t0:.4f}s")
except Exception as e:
    print(f"MPS crop error: {e}")
