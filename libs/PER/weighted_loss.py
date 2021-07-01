import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def weighted_loss(input, target, weights, beta=1):
        max_weight = max(weights)
        weights_ = torch.stack([weight/max_weight for weight in weights])
        batch_loss = (torch.abs(input - target) < beta).float() * 0.5 * (input - target)**2 / beta + \
            (torch.abs(input - target) >= beta).float() * ((torch.abs(input - target) - 0.5) * beta)
        weighted_batch_loss = weights_ * batch_loss
        #weighted_batch_loss = 1 * batch_loss # for testing purposes 
        weighted_loss = 1/len(batch_loss) * weighted_batch_loss.sum()
        return weighted_loss
