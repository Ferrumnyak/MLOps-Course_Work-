import torch

available = torch.cuda.is_available()
print("CUDA/ROCm available:", available)

# Дополнительно можно проверить устройство
if available:
    print("Device name:", torch.cuda.get_device_name(0))
