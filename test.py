from models import *
from dataloader import SegDataSet
from color_class import class2color, color2class
from torch.utils.data import DataLoader
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as tfs

MODEL_SAVE_PATH = "./data/codec_seg.pt"

def test():
    data_loader = DataLoader(dataset=SegDataSet(224, 224, False), batch_size=1, shuffle=True, num_workers=1)
    use_cuda = torch.cuda.is_available()
    #device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device("cpu")
    model = CodecNet13(21)
    if os.path.exists(MODEL_SAVE_PATH):
        state = torch.load(MODEL_SAVE_PATH)
        model.load_state_dict(state['net'])
    else:
        print("no model file error")
        return
    model.to(device)

    transform = tfs.Compose([tfs.ToPILImage()])
    for i_batch, sample_batched in enumerate(data_loader):
        img_tensor = sample_batched[0].to(device)
        label_tensor = sample_batched[1].to(device)
        output = model(img_tensor)
        predict_tensor = class2color(output)
        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)  # 开启一个窗口，同时设置大小，分辨率
        ax1 = fig.add_subplot(1, 3, 1)  # 通过fig添加子图，参数：行数，列数，第几个。
        ax2 = fig.add_subplot(1, 3, 2)  # 通过fig添加子图，参数：行数，列数，第几个。
        ax3 = fig.add_subplot(1, 3, 3)  # 通过fig添加子图，参数：行数，列数，第几个。
        ax1.imshow(transform(img_tensor[0]))
        ax3.imshow(transform(predict_tensor[0]))
        plt.show()
        plt.close()

if __name__=='__main__':
    test()