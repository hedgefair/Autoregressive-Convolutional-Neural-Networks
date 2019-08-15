import torch
import torch.nn as nn

def weights_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
