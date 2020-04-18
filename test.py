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
    data_loader = DataLoader(dataset=SegDataSet(224, 224, True), batch_size=args.batch, shuffle=True, num_workers=8)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = CodecNet13()
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    else:
        print("no model file error")
        return
    model.to(device)

    transform = tfs.Compose([tfs.ToPILImage()])
    for i_batch, sample_batched in enumerate(data_loader):
        img_tensor = sample_batched["img"].to(device)
        label_tensor = sample_batched["label"]
        label_tensor = color2class(label_tensor.int()).to(device)
        output = model(img_tensor)
        color_tensor = class2color(output)
        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)  # 开启一个窗口，同时设置大小，分辨率
        ax1 = fig.add_subplot(1, 2, 1)  # 通过fig添加子图，参数：行数，列数，第几个。
        ax2 = fig.add_subplot(1, 2, 2)  # 通过fig添加子图，参数：行数，列数，第几个。
        ax1.imshow(transform(sample_batched['img'][0]))
        ax2.imshow(transform(sample_batched['label'][0]))
        print("batch index ", i_batch)
        plt.show()
        plt.close()