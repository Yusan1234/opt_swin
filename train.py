import torch
import torch.nn as nn
import torch.optim as optim
from timm.models import create_model
from torch.profiler import profile, record_function, ProfilerActivity
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from timm.models import create_model
from torch.profiler import profile, record_function, ProfilerActivity

def train_swin(batch_size=32, image_size=224, epochs=1, device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=1000).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    train_data = CIFAR10(root='.', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=True,
            profile_memory=True,
            record_shapes=True
        ) as prof:
            for step, (x, y) in enumerate(train_loader):
                if step > 10: break
                optimizer.zero_grad()
                x, y = x.to(device, non_blocking=True), y.to(device)
                with record_function("forward"):
                    pred = model(x)
                    loss = loss_fn(pred, y)
                with record_function("backward"):
                    loss.backward()
                    optimizer.step()
                    
                prof.step()

    return prof