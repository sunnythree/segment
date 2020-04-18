import torch
import torch.nn.functional as F

a = torch.ones((3,3))
b = torch.ones((3,3))
c = (a+b)**2
print(c.sum())
