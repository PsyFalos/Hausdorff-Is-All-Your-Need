import glob
import numpy as np
import torch
import os
import cv2
from model.unet_model import UNet

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('best_model.pth', map_location=device))
    # 测试模式
    net.eval()

    # 确定结果保存的文件夹
    # 查找已有的测试结果文件夹
    result_folders = glob.glob('data/test_result_*')
    # 确定新的文件夹编号
    if not result_folders:
        folder_num = 1
    else:
        folder_nums = [int(folder.split('_')[-1]) for folder in result_folders]
        folder_num = max(folder_nums) + 1

    # 创建新的结果文件夹
    result_folder = f'data/test_result_{folder_num:02d}'
    os.makedirs(result_folder, exist_ok=True)
    print(f"预测结果将保存到文件夹: {result_folder}")

    # 读取所有图片路径
    tests_path = glob.glob('data/test/*.png')
    # 遍历所有图片
    for test_path in tests_path:
        # 提取文件名（不含路径和扩展名）
        file_name = os.path.basename(test_path).split('.')[0]
        # 设置保存路径
        save_res_path = os.path.join(result_folder, f'{file_name}_res.png')

        # 读取图片
        img = cv2.imread(test_path)
        # 转为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 转为batch为1，通道为1，大小为512*512的数组
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
        pred = net(img_tensor)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        # 保存图片
        cv2.imwrite(save_res_path, pred)

    print(f"预测完成，共处理 {len(tests_path)} 张图片")