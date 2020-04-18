from models import *
from summary import writer
from dataloader import SegDataSet
from color_class import class2color, color2class, color2class2
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
    parser.add_argument('--gama', "-g", type=float, default=0.9, help='train gama')
    parser.add_argument('--step', "-s", type=int, default=20, help='train step')
    parser.add_argument('--batch', "-b", type=int, default=1, help='train batch')
    parser.add_argument('--epoes', "-e", type=int, default=30, help='train epoes')
    parser.add_argument('--lr', "-l", type=float, default=0.001, help='learn rate')
    parser.add_argument('--pretrained', "-p", type=bool, default=False, help='prepare trained')
    return parser.parse_args()

def train(args):
    data_loader = DataLoader(dataset=SegDataSet(96, 96, True), batch_size=args.batch, shuffle=True, num_workers=8)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = CodecNet13()
    writer.add_graph(model, torch.zeros((1, 3, 96, 96)))
    if args.pretrained and os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gama)
    train_loss = 0
    loss_func = torch.nn.NLLLoss()
    for i in range(args.epoes):
        model.train()
        for i_batch, sample_batched in enumerate(data_loader):
            optimizer.zero_grad()
            img_tensor = sample_batched["img"].to(device)
            label_tensor = sample_batched["label"]
            label_tensor = color2class2(label_tensor).to(device)
            output = model(img_tensor)
            loss = loss_func(output, label_tensor)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            progress_bar(i_batch, len(data_loader), 'loss: '+str(train_loss))
            writer.add_scalar("loss", train_loss, global_step=i*args.batch+i_batch)
            if i_batch % 10 == 0:
                writer.add_image(str(i_batch), class2color(output)[0])
        scheduler.step()

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    writer.close()

if __name__=='__main__':
    train(parse_args())

