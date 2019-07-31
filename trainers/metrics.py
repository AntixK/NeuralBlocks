
# Binary Classification Metrics
def precision():
    pass

def recall():
    pass

def F1_score():
    pass

# Multi-class classification Metrics
def accuracy(outputs, targets):
    _, predicted = outputs.max(1)
    total = targets.size(0)
    correct = predicted.eq(targets).sum().item()
    return round(100. * correct / total, 3)


def top_K_accuracy(outputs, targets, k=3):
    total = targets.size(0)
    _, predicted = outputs.topk(k=k, dim=1)
    predicted =  predicted.t()
    targets = targets.view(1,-1).expand_as(predicted)
    correct = predicted.eq(targets).sum().item()
    return round(100. * correct / total, 3)

def top_3_accuracy(outputs, targets):
    return top_K_accuracy(outputs, targets, k=3)

def top_5_accuracy(outputs, targets):
    return top_K_accuracy(outputs, targets, k=5)

# Regression metrics
def RMSE():
    pass

def MSE():
    pass

def MSE():
    pass


if __name__ == "__main__":
    import torch
    torch.manual_seed(345)
    i = torch.randn(3,5)
    # print(i)
    o = torch.Tensor([4,2,4]).long()
    top_K_accuracy(i, o)
