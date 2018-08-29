import torch

a = torch.Tensor([1, 2, 3, 4, 5, 6, 7])
# mask = torch.lt(a, 4)
mask = a < 4
print(mask)

b = torch.masked_select(a, mask)
print(b)

c = a[mask]
print(c)

a = torch.Tensor([[8, 2], [2, 4], [5, 6], [7, 8]])
mask = a[:, 0] < 6
print(mask)
print(a[mask])
