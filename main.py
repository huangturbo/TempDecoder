import time
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch
from collections import OrderedDict

from models.tokshift import VideoNet

from dataloaders.hmdb_dataset import HMDBDataset

import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
torch.manual_seed(97)

class LabelSmoothingCrossEntropy(nn.Module):
	""" 
	NLL loss with label smoothing.
	"""
	def __init__(self, smoothing=0.1):
		"""
		Constructor for the LabelSmoothing module.
		:param smoothing: label smoothing factor
		"""
		super(LabelSmoothingCrossEntropy, self).__init__()								
		assert smoothing < 1.0 
		self.smoothing = smoothing
		self.confidence = 1. - smoothing

	def forward(self, x, target):
		logprobs = F.log_softmax(x, dim=-1)
		nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
		nll_loss = nll_loss.squeeze(1)
		smooth_loss = -logprobs.mean(dim=-1)
		loss = self.confidence * nll_loss + self.smoothing * smooth_loss
		return loss.mean()

def train_iter(args, model, optimz, data_load):
    samples = len(data_load.dataset)

    model.train()
    
    loss_fun = LabelSmoothingCrossEntropy()
    optimz.zero_grad()
    for i, (data, target) in enumerate(data_load):
        data = data.to(device)
        target = target.to(device)

        preds = model(data)

        loss = loss_fun(preds, target)
        loss = loss / args.accumulation_steps     
        loss.backward()

        if i % args.accumulation_steps == 0:
            optimz.step()       
            optimz.zero_grad()
        if i % 128 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_load)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item())) 

            
 
def evaluate_iter(model, data_load):
    
    model.eval()
    
    samples = len(data_load.dataset)
    csamp = 0

    with torch.no_grad():
        for data, target in data_load:
            data = data.to(device)
            target = target.to(device)

            preds = model(data)

            _, pred = torch.max(preds, dim=1)
            csamp += pred.eq(target).sum()
    
    print('\nAccuracy:' + '{:5}'.format(csamp) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(100.0 * csamp / samples) + '%)\n')

    return csamp / samples

def train(args):
    if args.dataset == 'hmdb51':
        num_classes = 51
        root_dir = './datasets/hmdb51/hmdb51_n_frames'
        test_list = './datasets/hmdb51/hmdb51_TrainTestlist/hmdb51_test.txt'
        train_list = './datasets/hmdb51/hmdb51_TrainTestlist/hmdb51_train.txt'
        
        test_dataset = HMDBDataset(root_dir, test_list, split='val', clip_len=args.num_frames)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        train_dataset = HMDBDataset(root_dir, train_list, split='train', clip_len=args.num_frames)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = VideoNet(num_class=num_classes,
                    with_decoder=args.with_decoder, 
                    vit_pretrain='',
                    sub_action_token_num=args.num_queries,
                    decoder_layer_num=args.decoder_layer_num,
                    with_spatial_conv=args.with_spatial_conv)
    new_state_dict = OrderedDict()
    model_state_dict = model.state_dict()
    pretrained_checkpoint = torch.load(args.pretrained_path, map_location=lambda storage, loc: storage)
    for k, v in pretrained_checkpoint['state_dict'].items():
        name = k[7:]
        if name in model_state_dict:
            if v.size() == model_state_dict[name].size():
                new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)

    model.to(device)

    optimz = optim.AdamW(model.parameters(), lr=args.lr, eps=1e-08, weight_decay=0.001)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimz, 'max')

    start_epoch = 1

    start_time = time.time()
    best_acc = 0.0
    for epoch in range(start_epoch, args.max_epoch + 1):
        print('Epoch:', epoch)
        train_iter(args, model, optimz, train_loader)
        val_acc = evaluate_iter(model, test_loader)
        scheduler.step(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimz.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
            }
            torch.save(checkpoint, f'{args.checkpoint_path}/saved_best_model.pth')
            print(f'saved {args.checkpoint_path}/saved_best_model.pth')
    
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')

def evaluate(args):
    checkpoint = torch.load(args.best_model_path, map_location=lambda storage, loc: storage)
    if args.dataset == 'hmdb51':
        num_classes = 51
        root_dir = './datasets/hmdb51/hmdb51_n_frames'
        test_list = './datasets/hmdb51/hmdb51_TrainTestlist/hmdb51_test.txt'
        
        test_dataset = HMDBDataset(root_dir, test_list, split='val', clip_len=args.num_frames)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = VideoNet(num_class=num_classes,
                    with_decoder=args.with_decoder, 
                    vit_pretrain='',
                    sub_action_token_num=args.num_queries,
                    decoder_layer_num=args.decoder_layer_num,
                    with_spatial_conv=args.with_spatial_conv)
    model.load_state_dict(checkpoint['state_dict'])
    evaluate_iter(model, test_loader)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Configuration of training")
    parser.add_argument("--batch_size", type = int, default=8)
    parser.add_argument("--num_frames", type = int, default=2, help="the number of frames for video clip samples")
    parser.add_argument("--accumulation_steps", type = int, default=8)
    parser.add_argument("--with_decoder", action='store_true', help="add decoder or not")
    parser.add_argument("--num_queries", type = int, default=6, help="the number of prediction queries")
    parser.add_argument("--decoder_layer_num", type = int, default=6, help="the number of layers in the decoder")
    parser.add_argument("--with_spatial_conv", action='store_true', help="whether to add cross-time spatial convolution module")
    parser.add_argument("--dataset", type = str, default='hmdb51')
    parser.add_argument("--pretrained_path", type = str, default='./pretrained/best_tokshift_B16x32x224_k400.pth', help="save path for pre-trained models on Kinetics400")
    parser.add_argument("--lr", type = float, default=0.00001, help="initial learning rate")
    parser.add_argument("--max_epoch", type = int, default=20, help="maximum epoch of training")
    parser.add_argument("--checkpoint_path", type = str, default='./checkpoints', help="save path for model checkpoints during training")
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)
