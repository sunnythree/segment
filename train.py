from models import *
from dataloader import SegDataSet
from color_class import class2color, color2class
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import argparse
import torch
import os
from utils import progress_bar

MODEL_SAVE_PATH = "./data/codec_seg.pt"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gama', type=float, default=0.9, help='train gama')
    parser.add_argument('--step', type=int, default=20, help='train step')
    parser.add_argument('--batch', type=int, default=20, help='train batch')
    parser.add_argument('--epoes', type=int, default=30, help='train epoes')
    parser.add_argument('--lr', type=float, default=0.001, help='learn rate')
    parser.add_argument('--pretrained', type=bool, default=False, help='prepare trained')
    return parser.parse_args()

def train(args):
    data_loader = DataLoader(dataset=SegDataSet(224, 224, True), batch_size=args.batch, shuffle=True)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = CodecNet13()
    if args.pretrained and os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gama)
    train_loss = 0
    for i in range(args.epoes):
        model.train()
        for i_batch, sample_batched in enumerate(data_loader):
            optimizer.zero_grad()
            img_tensor = sample_batched["img"].to(device)
            label_tensor = sample_batched["label"]
            label_tensor = color2class(label_tensor.int()).to(device)
            output = model(img_tensor)
            loss = F.smooth_l1_loss(output, label_tensor)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            progress_bar(i_batch, len(data_loader), 'Loss: %.6f , lr: %f' % ((train_loss/(i_batch+1)), scheduler.get_last_lr()[0]))

        scheduler.step()

    torch.save(model.state_dict(), MODEL_SAVE_PATH)

if __name__=='__main__':
    train(parse_args())

