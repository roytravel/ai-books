import torch

# GPU 정보 확인
print (torch.cuda.get_device_name(0))

# CUDA 사용 가능 여부 확인 (Compute Unified Device Architecture)
print (torch.cuda.is_available())

# torch 버전 확인
print (torch.__version__)
