import torch

A= torch.tensor([[1, 2], [3, 4]])
B= torch.tensor([[5, 6], [7, 8]])
result = torch.mm(A, B)
print(result)

result2 = torch.matmul(A, B)
print(result2)