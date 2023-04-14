import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
import time
import math
from torch.autograd import Variable
import numpy as np

import random
import numpy as np
import os

# 可直接调用此函数
def set_seed(seed=0):
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
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 这部分不要动，官方给的。。。
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def tr_epoch(model, data_loader, criterion1, criterionB, criterionBw, optimizer, args):
    # training-----------------------------
    model.train()
    loss_value1 = 0.
    loss_valueBw = 0.
    loss_valueB = 0.
    for i_batch, sample_batch in enumerate(data_loader):
        Volume = Variable(sample_batch['Volume']).cuda(args.device_ids)
        labels = Variable(sample_batch['label']).long().cuda(args.device_ids)
        t = -1
        Bw,B,outputs = model(Volume,t)
        loss1 = criterion1(outputs, labels)
        lossBw = criterionBw(Bw)
        lossB = criterionB(B)
        loss = loss1+lossBw+lossB
        loss_value1 += loss1
        loss_valueBw += lossBw
        loss_valueB += lossB

        if (i_batch+1)>int(np.floor(len(data_loader.dataset)/8))*8:
            loss = loss/(len(data_loader.dataset)-int(np.floor(len(data_loader.dataset)/8))*8)
            loss.backward()
            if (i_batch+1)==len(data_loader.dataset):
                optimizer.step()
                optimizer.zero_grad()
        else:
            loss = loss/8
            loss.backward()
            if (i_batch+1)%8==0:
                optimizer.step()
                optimizer.zero_grad()

    print('epoch Loss1: {:.6f}'.format(float(loss_value1.data)/(i_batch+1)), 'epoch LossBw: {:.6f}'.format(float(loss_valueBw.data)/(i_batch+1)), 'epoch LossB: {:.6f}'.format(float(loss_valueB.data)/(i_batch+1)))
    with open('./logtrain.txt', 'a') as out_file:
        out_file.write('epoch Loss1:{0},epoch LossBw:{1},epoch LossB:{2}'.format(float(loss_value1.data)/(i_batch+1),float(loss_valueBw.data)/(i_batch+1),float(loss_valueB.data)/(i_batch+1))+'\n')

def ts_epoch(model, data_loader, criterion1, criterionB, criterionBw, args):
    model.eval()
    count_correct = 0.
    loss_value1 = 0.
    loss_valueBw = 0.
    loss_valueB = 0.
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    predlist = '2'
    labelslist = '2'
    with torch.no_grad():
        for i_batch, sample_batch in enumerate(data_loader):
            Volume = Variable(sample_batch['Volume']).cuda(args.device_ids)
            labels = Variable(sample_batch['label']).long().cuda(args.device_ids)
            t = i_batch + 1
            Bw,B,outputs = model(Volume,t)

            loss1 = criterion1(outputs, labels)
            lossBw = criterionBw(Bw)
            lossB = criterionB(B)
            loss_value1 += loss1
            loss_valueBw += lossBw
            loss_valueB += lossB

            _,pred = torch.max(outputs, 1)


            if int(pred) == 1 and int(labels) == 1:
                TP = TP + 1
            elif int(pred) == 1 and int(labels) == 0:
                FP = FP + 1
            elif int(pred) == 0 and int(labels) == 0:
                TN = TN + 1
            elif int(pred) == 0 and int(labels) == 1:
                FN = FN + 1
            count_correct += torch.sum(pred == labels)
            predlist += str(int(pred))
            labelslist += str(int(labels))

        print('TP')
        print(TP)
        print('FP')
        print(FP)
        print('TN')
        print(TN)
        print('FN')
        print(FN)
        with open('result.txt', 'a') as f:
            f.write(labelslist + '\n')
            f.write(predlist + '\n')
            f.write('count_correct is:'+ str(float(count_correct)) + '\n')
            f.write('dataset is:'+str(len(data_loader.dataset))+'\n')
        print('Test Loss1: {:.6f}'.format(float(loss_value1.data)/(i_batch+1)),'epoch LossBw: {:.6f}'.format(float(loss_valueBw.data)/(i_batch+1)), 'epoch LossB: {:.6f}'.format(float(loss_valueB.data)/(i_batch+1)))
        #print('Acc is:', float(count_correct) / len(data_loader.dataset))
        print('count_correct is:', float(count_correct))
        print('dataset is:', len(data_loader.dataset))
        with open('./logtest.txt', 'a') as out_file:
            #out_file.write('Test Loss1:{0},epoch LossBw:{1},epoch LossB:{2},acc is:{3}'.format(float(loss_value1.data)/(i_batch+1),float(loss_valueBw.data)/(i_batch+1),float(loss_valueB.data)/(i_batch+1),float(count_correct) / len(data_loader.dataset))+'\n')
            out_file.write('Test Loss1:{0},epoch LossBw:{1},epoch LossB:{2},acc is:{3}'.format(float(loss_value1.data) / (i_batch + 1), float(loss_valueBw.data) / (i_batch + 1),float(loss_valueB.data) / (i_batch + 1), float(count_correct)) + '\n')
    #return float(count_correct) / len(data_loader.dataset)
    return float(count_correct)