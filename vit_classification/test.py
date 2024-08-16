import torch

c = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])
print(c.unfold(1, 2, 2).unfold(2, 1, 1))
print(c.unfold(1, 2, 2))