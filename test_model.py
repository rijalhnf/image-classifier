import torch
import time
from utils import get_device

def test_gpu_performance():
    device = get_device()
    print(f"Testing performance on: {device}")
    
    # Create random tensors
    size = 5000
    
    start_time = time.time()
    
    # Create tensors on the device
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Matrix multiplication
    c = torch.matmul(a, b)
    
    # Force completion of operations
    if device != torch.device("cpu"):
        torch.cuda.synchronize() if device == torch.device("cuda") else torch.mps.synchronize()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"Operation completed in {elapsed:.4f} seconds")
    print(f"Shape of result: {c.shape}")

if __name__ == "__main__":
    test_gpu_performance()