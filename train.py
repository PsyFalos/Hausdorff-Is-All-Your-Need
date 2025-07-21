from model.unet_model import UNet
from utils.dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
import numpy as np
import surface_distance as surf_metrics  # 需要安装：pip install surface-distance

# 定义Hausdorff距离损失函数
class HausdorffDistanceLoss(nn.Module):
    """
    基于surface-distance库实现的Hausdorff距离损失函数
    """

    def __init__(self, percentile=95):
        super(HausdorffDistanceLoss, self).__init__()
        self.percentile = percentile

    def forward(self, pred, target):
        # 将预测结果转换为概率值
        pred_sigmoid = torch.sigmoid(pred)

        # 将张量转换为二值图（阈值0.5）
        pred_binary = (pred_sigmoid > 0.5).float()

        # 初始化损失
        batch_size = pred.shape[0]
        loss = torch.tensor(0.0, device=pred.device)

        # 对每个批次样本计算Hausdorff距离
        for i in range(batch_size):
            pred_np = pred_binary[i, 0].detach().cpu().numpy().astype(bool)
            target_np = target[i, 0].detach().cpu().numpy().astype(bool)

            # 使用surface_distance库计算hausdorff距离
            surface_distances = surf_metrics.compute_surface_distances(
                target_np, pred_np, spacing_mm=(1.0, 1.0))

            # 计算HD95
            hd_dist = surf_metrics.compute_robust_hausdorff(surface_distances, self.percentile)

            # 累加到损失中
            loss += torch.tensor(hd_dist, device=pred.device)

        # 返回平均损失
        return loss / batch_size

# 定义组合损失函数
class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.7, hausdorff_weight=0.3):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.hausdorff_loss = HausdorffDistanceLoss()
        self.bce_weight = bce_weight
        self.hausdorff_weight = hausdorff_weight

    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        hausdorff = self.hausdorff_loss(pred, target)

        # 归一化hausdorff损失，防止数值过大
        hausdorff = torch.clamp(hausdorff / 100.0, 0, 1.0)

        return self.bce_weight * bce + self.hausdorff_weight * hausdorff

def train_net(net, device, data_path, epochs=40, batch_size=2, lr=0.00002):
    # 加载训练集
    isbi_dataset = ISBI_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 定义组合Loss算法：BCE + Hausdorff
    criterion = CombinedLoss(bce_weight=0.7, hausdorff_weight=0.3)
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        for image, label in train_loader:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(image)
            # 计算loss
            loss = criterion(pred, label)
            print('Loss/train', loss.item())
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')
            # 更新参数
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道1，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path = "data/train/"
    train_net(net, device, data_path)