import torch

# a = torch.Tensor([[1, 2, 3], [3, 4, 5]])
# b = a.unsqueeze(1)
# print(b)
# print(b.shape)

# a = torch.Tensor([1, 2, 3, 4, 6])
# mask = torch.lt(a, 3)
# print(mask)
# b = torch.nonzero(mask)
# print(b)

# a = torch.Tensor([1, 2, 3])
# b = torch.Tensor([4, 5, 6])
# c = torch.cat([a, b])
# print(c)
# d = torch.stack([a, b])
# print(d)

a = torch.Tensor([[1, 2], [3, 4], [5, 6]])
b = torch.Tensor([[5, 6], [7, 8], [4, 5]])
c = torch.cat([a, b], dim=1)
print(c)
print(c.shape)
d = torch.stack([a, b], dim=2)
print(d)
print(d.shape)
