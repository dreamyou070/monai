import os
import torch.nn.functional as F
import torch

logits = torch.tensor([[-0.6575, -0.7301]])
softmax_logits = F.softmax(logits)
log_s = torch.log(softmax_logits)
print(f'log_s = {log_s}')

log_probs = F.log_softmax(logits, dim=-1)
print(f'log_probs : {log_probs.shape}')
#selected = log_probs[range(len(logits)), y.view(-1)]