import os
log_probs = F.log_softmax(logits, dim=-1)
print(f'log_probs : {log_probs.shape}')
selected = log_probs[range(len(logits)), y.view(-1)]