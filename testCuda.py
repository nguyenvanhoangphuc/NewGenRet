import torch

if torch.cuda.is_available(): 
    dev = "cuda:0" 
else: 
    dev = "cpu" 
print(dev)
device = torch.device(dev) 
print(device)
a = torch.zeros(4,3) 
print(type(a))
print(a.device)
a = a.to(device)
import time
time.sleep(10)
print(a.device)