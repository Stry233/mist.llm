import torch.nn.functional as F

def cross_difference_loss(submodel_outputs):
    num_models = len(submodel_outputs)
    total_loss = 0.0
    for i in range(num_models):
        for j in range(i + 1, num_models):
            diff = F.mse_loss(submodel_outputs[i], submodel_outputs[j])
            total_loss += diff
    return total_loss / (num_models * (num_models - 1) / 2)
