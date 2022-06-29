import torch
import myConv1D

InputCh = 3
InputShape = 10

KeNum = 5
KeSize = 3
Stride = 2

KernelParameter = torch.ones(KeNum, InputCh, KeSize)
InputData = torch.rand(1, InputCh, InputShape)

res = myConv1D.forward(InputData, KernelParameter, Stride)

print(KernelParameter)
print(InputData)

print(res)
