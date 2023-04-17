from __future__ import print_function
from __future__ import division

# 可直接调用此函数
def set_seed(seed=1):
    print('seed = {}'.format(seed))
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 使用lstm需要添加下述环境变量为:16:8，如果cuda版本为10.2，去百度一下应该将环境变量设为多少。
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = True
    # use_deterministic_algorithms用于自查自己的代码是否包含不确定的算法，报错说明有，根据报错位置查询并替代该处的算法。1.8之前的版本好像没有此方法。
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 这部分不要动，官方给的。。。
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import torchvision.models as models

###Data require
import argparse
from datasets.dataset import VolumeDataset
from datasets.dataset import VolumeDatasetTest
from datasets.transforms import *
from torch.utils.data import DataLoader

# ###Model require
from models import resnet
from util import tr_epoch, ts_epoch


set_seed()

parser = argparse.ArgumentParser('Resnets')
parser.add_argument('--seed', type=int, default=1)

# ========================= Data Configs ==========================
parser.add_argument('--data_root_train', type=str, default='')
parser.add_argument('--list_file_train', type=str, default='./micro/train21.txt')
parser.add_argument('--data_root_test', type=str, default='')
parser.add_argument('--list_file_test', type=str, default='./micro/test21.txt')
parser.add_argument('--modality', type=str, default='Gray', help='RGB | Gray')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=16)

# ========================= Model Configs ==========================
parser.add_argument('--premodel', default='', type=str, help='Pretrained model (.pth)')
#parser.add_argument('--premodel', default='XXX/epoch100.pt', type=str, help='Pretrained model (.pth)')
parser.add_argument('--num_classes', default=2, type=int, help='Number of classes')
parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
parser.set_defaults(no_cuda=False)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--PenaltyBw', type=float, default=1)
parser.add_argument('--PenaltyB', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--device_ids',  type=int, default=0)

# ========================= Model Save ==========================
parser.add_argument('--save_path', type=str, default='./pt')
parser.add_argument('--checkpoint_path', type=str, default='')

args = parser.parse_args()



###Data read

test_dataset = VolumeDatasetTest(data_root=args.data_root_test, list_file_root=args.list_file_test, modality=args.modality,
                            transform=torchvision.transforms.Compose([
                                GroupScale((128,128)),
                                ToTorchFormatTensor(div=True),
                            ]),
)
test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=args.num_workers,drop_last=False)


# # ###Model
model = resnet.resnet18(pretrained=False, num_classes=args.num_classes)
if args.premodel:
    pretrained_dict = torch.load(args.premodel)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
    missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
    print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
    print('miss matched params:',missed_params)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
if not args.no_cuda:
    model = model.cuda(args.device_ids)
#print(model)


# ###Hyperparam
criterion1 = nn.CrossEntropyLoss()
if not args.no_cuda:
    criterion1 = criterion1.cuda(args.device_ids)
optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch/5, eta_min=1e-8, last_epoch=-1)

class BwLoss(nn.Module):
    def __init__(self):
        super(BwLoss, self).__init__()
    def forward(self, Bw):
        loss_Bw = 0.0
        for i in range(Bw.shape[0]):
            temp = Bw[i,:]
            loss_Bw += 2.0-(torch.mean(temp[temp>torch.mean(temp)])-torch.mean(temp[temp<torch.mean(temp)]))
        return  args.PenaltyBw*loss_Bw/Bw.shape[0]
criterionBw = BwLoss()
if not args.no_cuda:
    criterionBw = criterionBw.cuda(args.device_ids)

class BLoss(nn.Module):
    def __init__(self):
        super(BLoss, self).__init__()
    def forward(self, B):
        loss_B = 0.0
        for i in range(B.shape[0]):
            loss_B += torch.max(torch.Tensor([0.0]).cuda(args.device_ids), torch.sum(B[i,:]) - torch.Tensor([1.0]).cuda(args.device_ids))
        return  args.PenaltyB*loss_B/B.shape[0]
criterionB = BLoss()
if not args.no_cuda:
    criterionB = criterionB.cuda(args.device_ids)

Acc_best = 0.0
for epoch in range(1,args.epoch+1):
    train_dataset = VolumeDataset(data_root=args.data_root_train, list_file_root=args.list_file_train,
                                  modality=args.modality,
                                  transform=torchvision.transforms.Compose([
                                      GroupScaleRandomCrop((144, 144), (128, 128)),
                                      ToTorchFormatTensor(div=True),
                                  ]),
                                  )
    # train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,drop_last=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               pin_memory=True,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               worker_init_fn=seed_worker)
    print('train_dataset ', len(train_dataset))
    print('epoch {}:'.format(epoch), 'lr is {}'.format(optimizer.param_groups[0]['lr']))
    ###train and test
    print("Training-------------------")
    tr_epoch(model=model, data_loader=train_loader, criterion1=criterion1, criterionB=criterionB, criterionBw=criterionBw, optimizer=optimizer, args=args)
    print("Testing====================")
    Acc = ts_epoch(model=model, data_loader=test_loader, criterion1=criterion1, criterionB=criterionB, criterionBw=criterionBw, args=args)
    scheduler.step(epoch)
    #save model
    if epoch>=10:
        if Acc>=Acc_best:
            Acc_best = Acc
            torch.save(model.state_dict(), args.save_path + '/' + 'epoch' + str(epoch) + '.pt')