from models import *
from summary import writer
from dataloader import SegDataSet, data_prefetcher
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
    parser.add_argument('--gama', "-g", type=float, default=0.9, help='train gama')
    parser.add_argument('--step', "-s", type=int, default=20, help='train step')
    parser.add_argument('--batch', "-b", type=int, default=10, help='train batch')
    parser.add_argument('--epoes', "-e", type=int, default=30, help='train epoes')
    parser.add_argument('--lr', "-l", type=float, default=0.001, help='learn rate')
    parser.add_argument('--pretrained', "-p", type=bool, default=False, help='prepare trained')
    return parser.parse_args()

def train(args):
    start_epoch = 0
    data_loader = DataLoader(dataset=SegDataSet(224, 224, True), batch_size=args.batch, shuffle=True, num_workers=16)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = CodecNet13(21)
    writer.add_graph(model, torch.zeros((1, 3, 224, 224)))
    if args.pretrained and os.path.exists(MODEL_SAVE_PATH):
        state = torch.load(MODEL_SAVE_PATH)
        model.load_state_dict(state['net'])
        start_epoch = state['epoch']
    model = torch.nn.DataParallel(model, device_ids=[0, 1])  # multi-GPU
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gama)
    train_loss = 0
    loss_func = torch.nn.NLLLoss().to(device)
    for epoch in range(start_epoch, start_epoch+args.epoes):
        model.train()
        prefetcher = data_prefetcher(data_loader)
        img_tensor, label_tensor = prefetcher.next()
        last_img_tensor = img_tensor
        last_label_tensor = label_tensor
        i_batch = 0
        while img_tensor is not None:
            last_img_tensor = img_tensor
            last_label_tensor = label_tensor
            optimizer.zero_grad()
            output = model(img_tensor)
            loss = loss_func(output, label_tensor)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            global_step = epoch*len(data_loader)+i_batch
            progress_bar(i_batch, len(data_loader), 'loss: %f, epeche: %d'%(train_loss, epoch))
            writer.add_scalar("loss", train_loss, global_step=global_step)
            img_tensor, label_tensor = prefetcher.next()
            i_batch += 1

        #save one pic and output
        if i_batch % 10 == 0:
            writer.add_image("img: "+str(epoch), last_img_tensor[0])
            writer.add_image("output: "+str(epoch), class2color(output)[0])
        scheduler.step()

    if not os.path.isdir('data'):
        os.mkdir('data')
    print('Saving..')
    state = {
        'net': model.module.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, MODEL_SAVE_PATH)
    writer.close()

if __name__=='__main__':
    train(parse_args())

