"""
训练器模块
"""

import os
import unet
import torch
import dataset
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchmetrics
from sklearn.metrics import jaccard_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np


# 训练器
class Trainer:

    def __init__(self, path, model, model_copy, img_save_path):
        """
        :rtype: object
        """
        super(Trainer, self).__init__()
        self.path = path
        self.model = model
        self.model_copy = model_copy
        self.img_save_path = img_save_path
        # 使用的设备，如果有cuda用cuda，没有就用cpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 网络,将网络
        self.net = unet.UNet().to(self.device)
        # 优化器，这里用的Adam，跑得快点
        self.opt = torch.optim.RMSprop(self.net.parameters(), lr=0.00001, weight_decay=1e-8, momentum=0.9)
        # self.opt = torch.optim.Adam(self.net.parameters())
        # 这里直接使用二分类交叉熵来训练，效果可能不那么好
        # 可以使用其他损失，比如DiceLoss、Focal
        self.loss_func = nn.BCEWithLogitsLoss()
        # 加载数据集,加载了DataLoader，实例化一下
        # 设备好，batch_size和num_workers可以给大点，如果报错就调低
        self.loader = DataLoader(dataset.Datasets(path), batch_size=1, shuffle=True, num_workers=2)

        # 判断是否存在模型
        if os.path.exists(self.model):
            self.net.load_state_dict(torch.load(model))
            print(f"Loaded{model}!")
        else:
            print("不存在模型！")
        # os.makedirs：创建文件夹
        os.makedirs(img_save_path, exist_ok=True)

    # 训练
    def train(self, stop_value):
        self.net.train()
        # 轮次
        best_iou = float('-inf')
        best_loss = float('inf')
        epoch = 1
        ax = []
        ay = []
        plt.ion()
        while True:
            for inputs, labels in tqdm(self.loader, desc=f"Epoch {epoch}/{stop_value}",
                                       ascii=True, total=len(self.loader)):
                self.opt.zero_grad()  # 清空计步
                # 图片和分割标签，将数据放入设备里
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # 输出生成的图像。通过前向将原图放入？out为输出的图
                out = self.net(inputs)
                #
                loss = self.loss_func(out, labels)

                print(f" Loss: {loss}")

                # 保存iou值最大的网络参数
                if best_loss > loss:
                    best_loss = loss
                    torch.save(self.net.state_dict(), self.model)
                    print("saved_model 已经保存 !")

                # 后向
                loss.backward()  # 反向计数
                self.opt.step()  # 跟进计步？

                # 输入的图像，取第一张
                x = inputs[0]
                # 生成的图像，取第一张
                x_ = out[0]
                # 标签的图像，取第一张
                y = labels[0]
                # 三张图，从第0轴拼接起来，再保存
                img = torch.stack([x, x_, y], dim=0)
                # 保存地址，除了地址还要给文件名称
                # save_image(img, f'{save_path}/{i}.png')
                save_image(img.cpu(), os.path.join(self.img_save_path, f"{epoch}.png"))
                print("image save successfully !")
            ax.append(epoch)
            ay.append(best_loss.cpu().detach().numpy())
            plt.plot(ax, ay, 'ro--')
            plt.ioff()
            plt.pause(0.1)
            print(f"\nEpoch: {epoch}/{stop_value}, Loss: {best_loss}")
            # print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')
            # 备份，每隔50 批次就保存一次
            if epoch % 50 == 0:
                # 将net的参数放入.weight_path为保存地址
                torch.save(self.net.state_dict(), self.model_copy.format(epoch, loss))
                print("model_copy 已经保存 !")
            if epoch > stop_value:
                break

            # 一轮循环结束
            epoch += 1
        plt.show()


if __name__ == '__main__':    # 路径改一下

    t = Trainer("./lung/training", r'./saved_model/saved_model.plt',
                r'./saved_model/copy/model_{}_{}.plt', r'./training_images')
    # path: object 数据集最一开始的地址
    # saved_model: object,
    # model_copy: object
    # img_save_path: object 拼接好的图片的保存地址
    t.train(300)  # 300对应def train(self, stop_value)的stop_value，即if epoch > stop_value:break，训练300轮

