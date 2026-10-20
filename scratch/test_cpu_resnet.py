import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import ResNet18

device = torch.device("cpu")
torch.set_num_threads(8)
print("Device:", device, "Threads:", torch.get_num_threads())

model = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

imgs = torch.randn(36000, 3, 32, 32)
targets = torch.randint(0, 10, (36000,))

dataset = TensorDataset(imgs, targets)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

t0 = time.time()
model.train()
for batch_imgs, batch_targets in loader:
    optimizer.zero_grad()
    outputs = model(batch_imgs)
    loss = criterion(outputs, batch_targets)
    loss.backward()
    optimizer.step()

print(f"1 Full ResNet18 Epoch Time on CPU: {time.time() - t0:.4f}s")
