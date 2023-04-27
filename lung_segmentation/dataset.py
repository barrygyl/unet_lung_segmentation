import os
import cv2
import torchvision

from torch.utils.data import Dataset
from torchvision.utils import save_image
from torch.autograd import Variable


# 创建Datasets类，继承了torch的Dataset类：from torch.utils.data import Dataset
# 所有子类应该重写__len__和__getitem__，前者提供了数据集的大小，后者支持整数索引，范围从0到len(self)
class Datasets(Dataset):

    def __init__(self, path):
        self.path = path
        # 语义分割需要的图片的图片和标签。合并文件路径
        # os.path.join(path, "images")：拼接path和文件夹的名字，如：D:/lung/images/,D:/lung/1st_manual/，拼接成了完整路径
        # os.listdir() 在上面拼接完成的完整路径获取标签索引的名字，即获取文件夹下面所有文件名
        self.name1 = os.listdir(os.path.join(path, "images"))
        self.name2 = os.listdir(os.path.join(path, "1st_manual"))
        # 本代码用于归一化。图片都需要transforms，即归一化
        self.trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    """ 
            os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
            参数:path -- 需要列出的目录路径
            返回值:返回指定路径下的文件和文件夹列表

            os.path.join()函数用于路径拼接文件路径，可以传入多个路径。
            从后往前看，会从第一个以”/”开头的参数开始拼接，之前的参数全部丢弃；
            以上一种情况为先。在上一种情况确保情况下，若出现”./”开头的参数，会从”./”开头的参数的前面参数全部保留；

            torchvision是pytorch的一个图形库，它服务于PyTorch深度学习框架的，主要用来构建计算机视觉模型。
            torchvision.transforms主要是用于常见的一些图形变换。
            torchvision.transforms.Compose()类。这个类的主要作用是串联多个图片变换的操作。
           """

    # 这个函数是为了让len()能够工作，使类表现得像一个列表，通过len来获取他的长度。
    def __len__(self):
        # 这里return的是文件名的数量。也就是数据集的数量
        return len(self.name1)

    # 简单的正方形转换，把图片和标签转为正方形
    # 图片会置于中央，两边会填充为黑色，不会失真
    def __trans__(self, img, size):  # size在下面函数定义
        h, w = img.shape[0:2]  # img.shape[:2] 取彩色图片的长、宽。
        # 需要的尺寸
        _w = _h = size
        # 不改变图像的宽高比例
        scale = min(_h / h, _w / w)
        h = int(h * scale)  # 改为int型
        w = int(w * scale)  # 改为int型
        # 缩放图像
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        # INTER_CUBIC	4x4像素邻域的双三次插值	16个采样点加权平均。如果要缩小图像，推荐使用INTER_AREA插值；如果要放大图像，INTER_CUBIC。

        # 上下左右分别要扩展的像素数
        top = (_h - h) // 2
        left = (_w - w) // 2
        bottom = _h - h - top
        right = _w - w - left
        # 生成一个新的填充过的图像，这里用纯黑色进行填充(0,0,0)
        new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return new_img

    """
    如果你想给你的图片设置边界框，就像一个相框一样的东西，你就可以使用cv2.copyMakeBorder()函数。但其在卷积操作、零填充等也得到了应用，并且可以用于一些数据增广操作。
    src ： 输入的图片，即本函数中的img
    top, bottom, left, right ：相应方向上的边框宽度
    borderType：定义要添加边框的类型，它可以是以下的一种：cv2.BORDER_CONSTANT：添加的边界框像素值为常数（需要额外再给定一个参数）
    value：如果borderType为cv2.BORDER_CONSTANT时需要填充的常数值。
    """

    def __getitem__(self, index):  # index为索引
        # 拿到的图片和标签
        name1 = self.name1[index]  # 获取对应下标的数据名字：xxx.png or xxx.jpg
        name2 = self.name2[index]
        # 图片和标签的路径
        img_path = [os.path.join(self.path, i) for i in ("images", "1st_manual")]
        # 读取原始图片和标签，并转RGB
        img_o = cv2.imread(os.path.join(img_path[0], name1))
        img_l = cv2.imread(os.path.join(img_path[1], name2))
        img_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)  # BGR转RGB

        # 转成网络需要的正方形，使用自定义的def __trans__
        img_o = self.__trans__(img_o, 256)  # 要改！！！！256不对。涉及到需要的尺寸
        img_l = self.__trans__(img_l, 256)  # 要改！！！！

        # 返回转成网络需要的正方形，且进行归一化
        return self.trans(img_o), self.trans(img_l)


if __name__ == '__main__':

    i = 1
    dataset = Datasets("E:/lung/training")  # 传入地址，给一个变量接收

    for a, b in dataset:
        print(i)
        # print(a.shape)
        # print(b.shape)
        save_image(a, "E:/lung/training/images/" + str(i) + ".jpg", nrow=1)
        save_image(b, "E:/lung/training/images/" + str(i) + ".png", nrow=1)
        i += 1
        # 训练集一共39张图片
        if i > 39:
            break
