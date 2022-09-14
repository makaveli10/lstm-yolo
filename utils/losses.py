import torch
import torch.nn as nn


def bbox_iou(box1, box2, xywh=True, eps=1e-7):
    """
    Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, 1), box2.chunk(4, 1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, 1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, 1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    return iou if iou != 0 else iou + eps


class MSELoss(nn.Module):
    """
    Mean Squared Error Loss.
    """
    def __init__(self, beta=0.5):
        super(MSELoss, self).__init__()
        self.beta = beta
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred, gt):
        return self.beta * self.mse_loss(pred, gt)


class IOULoss(nn.Module):
    """
    Intersection Over Union Loss.
    """
    def __init__(self, alpha=1):
        super(IOULoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, preds, gts):
        total_iou = 0.0
        for i in range(len(preds)):
            total_iou += torch.log(bbox_iou(preds[i].reshape(1, -1), gts[i].reshape(1, -1)))
        iou_loss = -self.alpha * (total_iou / len(preds))
        return iou_loss


class ComputeLSTMLoss(nn.Module):
    def __init__(self):
        super(ComputeLSTMLoss, self).__init__()
        self.iou_loss = IOULoss()
        self.mse_loss = MSELoss()
    
    def forward(self, pred, gt, training=True):
        iou_loss = self.iou_loss(pred, gt)
        mse_loss = self.mse_loss(pred, gt)

        total_loss = iou_loss + mse_loss

        if training:
            loss_stats = {
                'train/total_loss': total_loss,
                'train/iou_loss': iou_loss,
                'train/mse_loss': mse_loss
            }
        else:
            loss_stats = {
                'val/total_loss': total_loss,
                'val/iou_loss': iou_loss,
                'val/mse_loss': mse_loss
            }
        return total_loss, loss_stats
