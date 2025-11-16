import torch
print(f"Доступно GPU: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Память: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")