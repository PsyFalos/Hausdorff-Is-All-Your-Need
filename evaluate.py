import os
import numpy as np
import cv2
import glob
from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate_segmentation():
    # 路径定义
    test_result_path = 'data/test_result_07'
    labels_path = 'data/test/labels'

    # 获取所有预测结果文件
    result_files = glob.glob(os.path.join(test_result_path, '*_res.png'))

    # 初始化评估指标
    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0

    count = 0

    for result_file in result_files:
        # 提取对应的文件名编号
        file_name = os.path.basename(result_file)
        img_idx = file_name.split('_')[0]  # 获取如"0"、"1"这样的编号

        # 对应的标签文件路径
        label_file = os.path.join(labels_path, f"{img_idx}.png")

        # 检查标签文件是否存在
        if not os.path.exists(label_file):
            print(f"警告：找不到标签文件 {label_file}")
            continue

        # 读取预测结果和标签
        pred = cv2.imread(result_file, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)

        # 确保图像尺寸一致
        if pred.shape != label.shape:
            print(f"警告：图像 {img_idx} 的预测结果和标签尺寸不一致")
            continue

        # 二值化处理
        pred = (pred > 127).astype(np.uint8)
        label = (label > 127).astype(np.uint8)

        # 计算指标
        intersection = np.logical_and(pred, label).sum()
        union = np.logical_or(pred, label).sum()

        # 避免除零错误
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union

        # 计算Dice系数
        if pred.sum() + label.sum() == 0:
            dice = 1.0
        else:
            dice = 2 * intersection / (pred.sum() + label.sum())

        # 将图像展平为1D数组以便使用sklearn计算precision和recall
        pred_flat = pred.flatten()
        label_flat = label.flatten()

        precision = precision_score(label_flat, pred_flat, zero_division=1)
        recall = recall_score(label_flat, pred_flat, zero_division=1)

        # 累加指标
        total_dice += dice
        total_iou += iou
        total_precision += precision
        total_recall += recall
        count += 1

        print(f"图像 {img_idx}: Dice={dice:.4f}, IoU={iou:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

    # 计算平均值
    if count > 0:
        avg_dice = total_dice / count
        avg_iou = total_iou / count
        avg_precision = total_precision / count
        avg_recall = total_recall / count

        print("\n评估结果汇总:")
        print(f"平均Dice系数: {avg_dice:.4f}")
        print(f"平均IoU: {avg_iou:.4f}")
        print(f"平均精确率: {avg_precision:.4f}")
        print(f"平均召回率: {avg_recall:.4f}")
    else:
        print("没有找到可评估的图像对")


if __name__ == "__main__":
    evaluate_segmentation()