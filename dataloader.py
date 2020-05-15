from torch.utils.data import DataLoader
from torchvision import transforms as tfs
from torch.utils.data import Dataset
import random
import math
from PIL import Image
import matplotlib.pyplot as plt
import torch
from color_class import color2class

PICS_PATH = "/home/javer/work/dataset/voc/VOCdevkit/VOC2007"
ORIGIN_PATH = "/JPEGImages/"
SEGMENT_PATH = "/SegmentationClass/"
TRAIN_FILE_PATH = "/ImageSets/Segmentation/train.txt"
VAL_FILE_PATH = "/ImageSets/Segmentation/val.txt"


class SegDataSet(Dataset):
    def __init__(self, orgin_size, label_size, is_train=True):
        self.is_train = is_train
        self.orgin_size = orgin_size
        self.label_size = label_size
        self.radio = label_size/orgin_size
        self.pic_strong = tfs.Compose([
            tfs.ColorJitter(0.5, 0.2, 0.2, 0.1),
            tfs.ToTensor()
        ])
        self.pic_image2tensor = tfs.Compose([
            tfs.ToTensor()
        ])
        self.pics = []
        self.imgs = []
        self.labels = []
        if is_train:
            for line in open(PICS_PATH+TRAIN_FILE_PATH):
                fname = line.replace('\n', '')
                self.pics.append(fname)
                # img
                img = Image.open(PICS_PATH + ORIGIN_PATH + fname + ".jpg")
                img, rand_p = pic_resize2square(img, self.orgin_size, None, True)
                self.imgs.append(img_tensor)

                # label
                label_img = Image.open(PICS_PATH + SEGMENT_PATH + fname + ".png")
                label_img, _ = pic_resize2square(label_img, self.label_size,
                                                 (int(rand_p[0] * self.radio), int(rand_p[1] * self.radio)))
                label_tensor = self.pic_image2tensor(label_img)
                label_tensor *= 255
                label_tensor = color2class(label_tensor)
                self.labels.append(label_tensor)
        else:
            for line in open(PICS_PATH+"/"+TRAIN_FILE_PATH):
                self.pics.append(line.replace('\n', ''))


    def __len__(self):
        return len(self.pics)

    def __getitem__(self, item):
        if self.is_train:
            return self.pic_strong(self.imgs[item]), self.labels[item]
        else:
            # img
            img = Image.open(PICS_PATH + ORIGIN_PATH + self.pics[item] + ".jpg")
            img, rand_p = pic_resize2square(img, self.orgin_size, None, True)

            # label
            label_img = Image.open(PICS_PATH + SEGMENT_PATH + self.pics[item] + ".png")
            label_img, _ = pic_resize2square(label_img, self.label_size,
                                             (int(rand_p[0] * self.radio), int(rand_p[1] * self.radio)))
            label_tensor = self.pic_image2tensor(label_img)
            label_tensor *= 255
            label_tensor = color2class(label_tensor)
            return img_tensor, label_tensor


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target

def pic_resize2square(img, des_size, rand_p=None, is_random=True):
    rows = img.height
    cols = img.width
    scale_rate = float(0)
    if rand_p is None:
        rand_x = int(0)
        rand_y = int(0)
    elif len(rand_p) != 2:
        print("rand_p required length is 2")
        return None
    new_rows = des_size
    new_cols = des_size
    if rows > cols:
        scale_rate = des_size/rows
        new_cols = math.ceil(cols*scale_rate)
        # print(rows, cols, new_rows, new_cols, scale_rate)
        if rand_p is None:
            if is_random:
                rand_x = random.randint(0, math.floor(new_rows - new_cols))
            else:
                rand_x = math.floor(new_rows - new_cols)
    elif cols > rows:
        scale_rate = des_size/cols
        new_rows = math.ceil(rows*scale_rate)
        # print(rows, cols, new_rows, new_cols, scale_rate)
        if rand_p is None:
            if is_random:
                rand_y = random.randint(0, math.floor(new_cols - new_rows))
            else:
                rand_y = math.floor(new_cols - new_rows)

    new_img = img.resize((new_cols, new_rows))
    scaled_img = Image.new("RGB", (des_size, des_size))
    if rand_p is None:
        scaled_img.paste(new_img, box=(rand_x, rand_y))
        return scaled_img, (rand_x, rand_y)
    else:
        scaled_img.paste(new_img, box=rand_p)
        return scaled_img, tuple(rand_p)


def test_tools():
    img = Image.open(PICS_PATH + ORIGIN_PATH + "2007_000032.jpg")
    figs = plt.subplot()
    scaled_img = pic_resize2square(img, 408, True)
    transform = tfs.Compose([
        tfs.ColorJitter(0.5, 0.2, 0.2, 0.1),
    ])
    scaled_img = transform(scaled_img)
    figs.imshow(scaled_img)
    plt.show()

def test_dataset():
    data_loader = DataLoader(dataset=SegDataSet(408, 204, True), batch_size=1, shuffle=True)
    transform = tfs.Compose([tfs.ToPILImage()])
    for i_batch, sample_batched in enumerate(data_loader):
        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)  # 开启一个窗口，同时设置大小，分辨率
        ax1 = fig.add_subplot(1, 2, 1)  # 通过fig添加子图，参数：行数，列数，第几个。
        ax2 = fig.add_subplot(1, 2, 2)  # 通过fig添加子图，参数：行数，列数，第几个。
        ax1.imshow(transform(sample_batched['img'][0]))
        ax2.imshow(transform(sample_batched['label'][0]))
        print("batch index ", i_batch)
        plt.show()
        plt.close()

#test_dataset()

