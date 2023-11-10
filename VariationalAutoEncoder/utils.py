import torch
import torch.nn as nn


class WeightedAsymmetricLoss(nn.Module):
    def __init__(self, eps=1e-8, disable_torch_grad=True):
        super(WeightedAsymmetricLoss, self).__init__()
        self.disable_torch_grad = disable_torch_grad
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.tensor([1.342985,1.689665,1.248091,0.561356,0.433044,0.435894,0.457287,0.541004,1.408054,0.722762,1.493145,1.666711]),requires_grad=False)    
        

    def forward(self, x, y):
        xs_pos = x
        xs_neg = 1 - x

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.disable_torch_grad:
            torch.set_grad_enabled(False)
        neg_weight = 1 - xs_neg
        if self.disable_torch_grad:
            torch.set_grad_enabled(True)
        loss = los_pos + neg_weight * los_neg
        loss = loss * self.weight.view(1,-1)
    
        # loss = loss.mean(dim=-1)
        return -loss.mean()


def MultiLabelF1(pred, target, threshold=0.5):
    num_class = pred.shape[1]
    f1_all = torch.ones(num_class, device=target.device)
    pred = pred > threshold
    pred = pred.long()
    for label in range(num_class):
        pred_label = pred[:, label]
        target_label = target[:, label]
        tp = torch.sum(pred_label * target_label)
        fp = torch.sum(pred_label * (1 - target_label))
        fn = torch.sum((1 - pred_label) * target_label)
        precision = tp / (tp + fp + 1e-20)
        recall = tp / (tp + fn + 1e-20)
        f1 = 2 * precision * recall / (precision + recall + 1e-20)
        f1_all[label] = f1
    return f1_all

if __name__ == "__main__":
    pred = torch.rand(10, 12)
    target = torch.randint(0, 2, (10, 12))
    loss = WeightedAsymmetricLoss()
    print(loss(pred, target))
    print(MultiLabelF1(pred, target))