import argparse
import numpy as np
import os
import torch
#from tensorboardX import SummaryWriter
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from evaluation import evaluate
from loss import InpaintingLoss
from net import MADFNet
from net import VGG16FeatureExtractor
from dataset import trainset
from dataset import testset
from util.io import load_ckpt
from util.io import save_ckpt
import opt
from data_parallel import DataParallel_withLoss
class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0


parser = argparse.ArgumentParser()
# training options
parser.add_argument('--train_root', type=str, default='')
parser.add_argument('--test_root', type=str, default='')
parser.add_argument('--mask_root', type=str, default='')
parser.add_argument('--save_dir', type=str, default='./output/snapshots/default')
parser.add_argument('--log_dir', type=str, default='./log/default/')
parser.add_argument('--log_file', type=str, default='log')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--max_iter', type=int, default=10000000)
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=20000)
parser.add_argument('--vis_interval', type=int, default=5000)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--image_size', type=int, default=512)
parser.add_argument('--load_size', type=int, default=572)
parser.add_argument('--resume', type=str)
parser.add_argument('--use_incremental_supervision', action='store_true')
parser.add_argument('--n_refinement_D', type=int, default=2)
parser.add_argument('--valid_weight', type=float, default=1.0)
parser.add_argument('--hole_weight', type=float, default=6.0)
parser.add_argument('--tv_weight', type=float, default=0.1)
parser.add_argument('--prc_weight', type=float, default=0.05)
parser.add_argument('--style_weight', type=float, default=120.0)
args = parser.parse_args()


torch.backends.cudnn.benchmark = True
device = torch.device('cuda')

if not os.path.exists(args.save_dir):
    os.makedirs('{:s}/images'.format(args.save_dir))
    os.makedirs('{:s}/ckpt'.format(args.save_dir))

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
size = (args.image_size, args.image_size)
img_tf = transforms.Compose(
    [transforms.Resize(size=args.load_size),
     transforms.RandomCrop(size=size),
     transforms.RandomHorizontalFlip(), 
     transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_tf = transforms.Compose(
    [transforms.Resize(size=size), transforms.RandomRotation(10), transforms.RandomHorizontalFlip(), transforms.ToTensor()])

dataset_train = trainset(args.train_root, args.mask_root, img_tf, mask_tf)
dataset_val = trainset(args.test_root, args.mask_root, img_tf, mask_tf)
test_img_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
test_mask_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])
iters_per_epoch = len(dataset_train) / args.batch_size
iterator_train = iter(data.DataLoader(
    dataset_train, batch_size=args.batch_size,
    sampler=InfiniteSampler(len(dataset_train)),
    num_workers=args.n_threads))
model = MADFNet(layer_size=7, args=args).to(device)
print(model)
lr = args.lr

start_iter = 0
gen_optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
criterion = InpaintingLoss(VGG16FeatureExtractor(), args).to(device)
if args.resume:
    start_iter = load_ckpt(
        args.resume, [('model', model)], [('gen_optimizer', gen_optimizer)])
    for param_group in gen_optimizer.param_groups:
        param_group['lr'] = lr
    with open(args.log_dir + '/' + args.log_file, 'a') as writer:
        writer.write('===================Starting from iter {:d}==================\n'.format(start_iter))
torch.backends.cudnn.benchmark = True
model = DataParallel_withLoss(model, VGG16FeatureExtractor(), args)

for i in tqdm(range(start_iter, args.max_iter)):
    model.train()

    image, mask, gt = [x.to(device) for x in next(iterator_train)]
    
    outputs, feats, comp_feats, feat_gt = model(image, mask, gt)
    loss_dict = criterion(image, mask, outputs, gt, comp_feats, feats, feat_gt)

    gen_loss = 0.0
    for key in loss_dict:
        coef = getattr(args, key+'_weight') 
        value = coef * loss_dict[key]
        gen_loss += value
    if (i + 1) % args.log_interval == 0:
        epoch = i / iters_per_epoch
        iters = i % iters_per_epoch
        log = "Epoch : %d iters : %d "%(epoch, iters)
        for key in loss_dict:
            value = loss_dict[key]
            log += "loss_{:s} : ".format(key)+"{:f} ".format(value.item())
        log += "\n"
        with open(args.log_dir + '/' + args.log_file, 'a') as writer:
            writer.write(log)

    del loss_dict
    
    
    gen_optimizer.zero_grad()
    gen_loss.backward()
    gen_optimizer.step()

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        save_ckpt('{:s}/ckpt/{:d}.pth'.format(args.save_dir, i + 1),
                  [('model', model)], [('gen_optimizer', gen_optimizer)], i + 1)

    if (i + 1) % args.vis_interval == 0:
        model.eval()
        evaluate(model, dataset_val, device,
                 '{:s}/images/test_{:d}.jpg'.format(args.save_dir, i + 1))

