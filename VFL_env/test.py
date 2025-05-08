import torch 

val = 0.12222333
test=torch.tensor(val,dtype=torch.float64)

res=test + 0.03

print(res.item())