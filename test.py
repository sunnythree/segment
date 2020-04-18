import torch.nn as nn
import torch

m = nn.Softmax(dim=0)
n = nn.Softmax(dim=1)
#k = nn.Softmax(dim=2)
input = torch.ones(4, 3)
print(input)
print(m(input))
print(n(input))
#print(k(input))