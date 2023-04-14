import torch
import torch.nn as nn
from torch.autograd import Function
import torch.utils.model_zoo as model_zoo




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




__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride, groups)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
#上面是残差网络代码


# class BinarizedF(Function):
#     def forward(self, input):
#         self.save_for_backward(input)
#         ones = torch.ones_like(input)
#         zeros = torch.zeros_like(input)
#         output = torch.where(input>0,ones,zeros)
#         return output
#     def backward(self, output_grad):
#         input, = self.saved_tensors
#         ones = torch.ones_like(input)
#         zeros = torch.zeros_like(input)
#         input_grad = output_grad*torch.where((0<input)&(input<1), ones, zeros)
#         return input_grad
# class BinarizedModule(nn.Module):
#   def __init__(self):
#     super(BinarizedModule, self).__init__()
#   def forward(self,input):
#     output =BinarizedF()(input)
#     return output

class BinarizedF(Function):
    def forward(self, input):
        self.save_for_backward(input)
        ones = torch.ones_like(input)
        zeros = torch.zeros_like(input)
        output = input.clone()
        for n in range(input.shape[0]):
            output[n,:] = torch.where(input[n,:]>=torch.mean(input[n,:]),ones,zeros)
        return output
    def backward(self, output_grad):
        input, = self.saved_tensors
        ones = torch.ones_like(input)
        zeros = torch.zeros_like(input)
        input_grad = output_grad.clone()
        for n in range(input.shape[0]):
            input_grad[n,:] = output_grad[n,:]*torch.where((1>torch.mean(input[n,:]))&(0<torch.mean(input[n,:])), ones, zeros)
        return input_grad
class BinarizedModule(nn.Module):
  def __init__(self):
    super(BinarizedModule, self).__init__()
  def forward(self,input):
    output =BinarizedF()(input)
    return output



class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        planes = [int(width_per_group * groups * 2 ** i) for i in range(4)]
        self.inplanes = planes[0]
        self.conv1 = nn.Conv2d(1, planes[0], kernel_size=5, stride=2, padding=2,
                               bias=False)
        self.bn1 = norm_layer(planes[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, planes[0], layers[0], groups=groups, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2, groups=groups, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2, groups=groups, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, planes[3], layers[3], stride=2, groups=groups, norm_layer=norm_layer)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.alpha = nn.Sequential(nn.Linear(512, 1),
                                   nn.Sigmoid())
        self.BinarizedModulev1 = BinarizedModule()

        self.gru = nn.GRU(512*1*1, 32, 2, bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(planes[3] * block.expansion * 16 // 8, 2)
        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                nn.init.orthogonal_(m.weight_ih_l0)
                nn.init.orthogonal_(m.weight_hh_l0)
                nn.init.uniform_(m.bias_ih_l0)
                nn.init.uniform_(m.bias_hh_l0)
                nn.init.orthogonal_(m.weight_ih_l1)
                nn.init.orthogonal_(m.weight_hh_l1)
                nn.init.uniform_(m.bias_ih_l1)
                nn.init.uniform_(m.bias_hh_l1)


        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, groups=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, Volume, t):

        Volume = Volume.permute(0, 2, 1, 3, 4).contiguous()   ###N C T H W  to  N T C H W
        x = Volume.view(-1, Volume.size(2), Volume.size(3), Volume.size(4))  ### N*T C H W
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) ### N*T 512 4 4

        if t==-1:
            # #关键帧提取
            # BX1 = self.avgpool(x).squeeze(3).squeeze(2) ### N*T 512
            # BX2 = BX1*self.alpha(BX1)### N*T 512
            # BXwhole = BX2.view(Volume.size(0),Volume.size(1),512).sum(dim=1)### N 512
            # #两者想乘
            # Bw = torch.matmul(nn.functional.normalize(BX1).view(Volume.size(0),Volume.size(1),512),nn.functional.normalize(BXwhole).unsqueeze(2)).squeeze(2)### N T
            # B = self.BinarizedModulev1(Bw) ### N T     0 1    二值化
            # #输出关键帧的个数
            # #print(B.sum().item())
            Bw = torch.Tensor([[0.98, 0.98, 0.98, 0.97, 0.97]]).cuda()
            B = torch.Tensor([[1., 1., 1., 0., 0.]]).cuda()
        else:
            Bw = torch.Tensor([[0.9888, 0.9764, 0.9878, 0.9742, 0.9882]]).cuda() #先掩盖一下，等好了再删掉

            # #SAMM宏表情的关键帧
            # num = 22-1  # 根据train和test更改值 从0开始
            # with open('macrokey.txt', 'r') as f:
            #     data = f.readlines()
            # with open('midmacrokey.txt', 'r') as f:
            #     data1 = f.readlines()
            # for i in range(0, 294374):
            #     if (data[i] == 'Selecting split: ' + str(num) + '\n'):
            #         i1 = i
            #         break
            #
            # num1 = num + 1
            # i2 = 294374
            # for i in range(0, 294374):
            #     if (data[i] == 'Selecting split: ' + str(num1) + '\n'):
            #         i2 = i
            #         break
            # amount = int((i2 - i1 - 1) / 2)  # 微表情的总数
            # # 得到无表情的初始位置
            # for i in range(0, 294374):
            #     if (data1[i] == 'Selecting split: ' + str(num) + '\n'):
            #         i3 = i
            #         break
            # # 判断输入的t是属于微表情还是无表情
            # if t <= amount:  # 属于微表情
            #     # print(data[t * 2 + i1 - 1])
            #     B = torch.Tensor([[float(data[t * 2 + i1][1:3]), float(data[t * 2 + i1][4:6]), float(data[t * 2 + i1][7:9]),
            #                        float(data[t * 2 + i1][10:12]), float(data[t * 2 + i1][13:15])]]).cuda()
            #     # print('B')
            #     # print(B)
            # else:  # 属于无表情
            #     # print(data1[(t - amount) * 2 + i3 - 1])
            #     B = torch.Tensor([[float(data1[(t - amount) * 2 + i3][1:3]), float(data1[(t - amount) * 2 + i3][4:6]),
            #                        float(data1[(t - amount) * 2 + i3][7:9]), float(data1[(t - amount) * 2 + i3][10:12]),
            #                        float(data1[(t - amount) * 2 + i3][13:15])]]).cuda()
            #     # print('B')
            #     # print(B)

            # #SAMM微表情的关键帧
            # num = 28-1  # 根据train和test更改值 从0开始
            # with open('microkey.txt', 'r') as f:
            #     data = f.readlines()
            # with open('midmicrokey.txt', 'r') as f:
            #     data1 = f.readlines()
            # for i in range(0, 22390):
            #     if (data[i] == 'Selecting split: ' + str(num) + '\n'):
            #         i1 = i
            #         break
            #
            # num1 = num + 1
            # i2 = 22390
            # for i in range(0, 22390):
            #     if (data[i] == 'Selecting split: ' + str(num1) + '\n'):
            #         i2 = i
            #         break
            # amount = int((i2 - i1 - 1) / 2)  # 微表情的总数
            # # 得到无表情的初始位置
            # for i in range(0, 22390):
            #     if (data1[i] == 'Selecting split: ' + str(num) + '\n'):
            #         i3 = i
            #         break
            # # 判断输入的t是属于微表情还是无表情
            # if t <= amount:  # 属于微表情
            #     # print(data[t * 2 + i1 - 1])
            #     B = torch.Tensor([[float(data[t * 2 + i1][1:3]), float(data[t * 2 + i1][4:6]), float(data[t * 2 + i1][7:9]),
            #                        float(data[t * 2 + i1][10:12]), float(data[t * 2 + i1][13:15])]]).cuda()
            #     # print('B')
            #     # print(B)
            # else:  # 属于无表情
            #     # print(data1[(t - amount) * 2 + i3 - 1])
            #     B = torch.Tensor([[float(data1[(t - amount) * 2 + i3][1:3]), float(data1[(t - amount) * 2 + i3][4:6]),
            #                        float(data1[(t - amount) * 2 + i3][7:9]), float(data1[(t - amount) * 2 + i3][10:12]),
            #                        float(data1[(t - amount) * 2 + i3][13:15])]]).cuda()
            #     # print('B')
            #     # print(B)

            # 宏表情的关键帧
            num = 1-1  # 根据train和test更改值
            with open('hbqkey.txt', 'r') as f:
                data = f.readlines()
            with open('wuhbqkey.txt', 'r') as f:
                data1 = f.readlines()
            for i in range(0, 18570):
                if (data[i] == 'Selecting split: ' + str(num) + '\n'):
                    i1 = i
                    break

            num1 = num + 1
            i2 = 18577
            for i in range(0, 18570):
                if (data[i] == 'Selecting split: ' + str(num1) + '\n'):
                    i2 = i
                    break
            amount = int((i2 - i1 - 1) / 2)  # 宏表情的总数
            # 得到无表情的初始位置
            for i in range(0, 18570):
                if (data1[i] == 'Selecting split: ' + str(num) + '\n'):
                    i3 = i
                    break
            # 判断输入的t是属于宏表情还是微表情
            if t <= amount:  # 属于宏表情
                # print(data[t * 2 + i1 - 1])
                B = torch.Tensor([[float(data[t * 2 + i1][1:3]), float(data[t * 2 + i1][4:6]),
                                   float(data[t * 2 + i1][7:9]), float(data[t * 2 + i1][10:12]),
                                   float(data[t * 2 + i1][13:15])]]).cuda()

            else:  # 属于无表情
                # print(data1[(t - amount) * 2 + i3 - 1])
                B = torch.Tensor([[float(data1[(t - amount) * 2 + i3][1:3]), float(data1[(t - amount) * 2 + i3][4:6]),
                                   float(data1[(t - amount) * 2 + i3][7:9]), float(data1[(t - amount) * 2 + i3][10:12]),
                                   float(data1[(t - amount) * 2 + i3][13:15])]]).cuda()


            # #微表情的关键帧
            # num = 11-1  # 根据train和test更改值
            # with open('wbqkey.txt', 'r') as f:
            #     data = f.readlines()
            # with open('wuwbqkey.txt', 'r') as f:
            #     data1 = f.readlines()
            # for i in range(0, 1043):
            #     if (data[i] == 'Selecting split: ' + str(num) + '\n'):
            #         i1 = i
            #         break
            #
            # num1 = num + 1
            # i2 = 1043
            # for i in range(0, 1043):
            #     if (data[i] == 'Selecting split: ' + str(num1) + '\n'):
            #         i2 = i
            #         break
            # amount = int((i2 - i1 - 1) / 2)  # 微表情的总数
            # # 得到无表情的初始位置
            # for i in range(0, 1043):
            #     if (data1[i] == 'Selecting split: ' + str(num) + '\n'):
            #         i3 = i
            #         break
            # # 判断输入的t是属于微表情还是无表情
            # if t <= amount:  # 属于微表情
            #     # print(data[t * 2 + i1 - 1])
            #     B = torch.Tensor([[float(data[t * 2 + i1][1:3]), float(data[t * 2 + i1][4:6]), float(data[t * 2 + i1][7:9]),
            #                        float(data[t * 2 + i1][10:12]), float(data[t * 2 + i1][13:15])]]).cuda()
            #     # print('B')
            #     # print(B)
            # else:  # 属于无表情
            #     # print(data1[(t - amount) * 2 + i3 - 1])
            #     B = torch.Tensor([[float(data1[(t - amount) * 2 + i3][1:3]), float(data1[(t - amount) * 2 + i3][4:6]),
            #                        float(data1[(t - amount) * 2 + i3][7:9]), float(data1[(t - amount) * 2 + i3][10:12]),
            #                        float(data1[(t - amount) * 2 + i3][13:15])]]).cuda()
            #     # print('B')
            #     # print(B)

        ###for N==1 so:
        x = x.view(Volume.size(0),Volume.size(1),512,4,4).mul(B.repeat(512,4,4,1,1).permute(3, 4, 0, 1, 2).contiguous()) ### N T 512 4 4
        xBO = torch.index_select(x, 1, (B.squeeze()==1).nonzero().squeeze()) ###N ? 512 4 4
        xBO = xBO.permute(0, 3, 4, 1, 2).contiguous() #N 4 4 ? 512
        xBO = xBO.view(-1, xBO.size(3), 512) #N*4*4 ? 512
        #print(xBO.size())
        out, _ = self.gru(xBO) #N*4*4 ? 64
        # out = out[:,-1,:] #N*4*4 1 64
        out = torch.mean(out,dim=1) #N*4*4 64
        out = out.view(-1, 4*4*64) #N 4*4*64
        out = self.fc(out) #N 4
        return Bw,B,out


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

    return model


def resnext50_32x4d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], groups=4, width_per_group=32, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnext101_32x8d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], groups=8, width_per_group=32, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model