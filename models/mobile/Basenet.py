import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .contextual_layer import ContextualModule
from torch.utils import model_zoo

class BR(nn.Module):
    def __init__(self, nOut):
        super(BR,self).__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)
    def forward(self, input):
        output = self.bn(input)
        output = self.act(output)
        return output
class CBR(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1,dilated = 1):
        super(CBR,self).__init__()
        padding = int((kSize - 1)/2)*dilated
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False, dilation=dilated)
        self.BR = BR(nOut)

    def forward(self, input):
        output = self.conv(input)
        output = self.BR(output)
        return output
class upBR(nn.Module):
    def __init__(self, nIn, nOut):
        super(upBR,self).__init__()
        self.conv = nn.ConvTranspose2d(nIn, nOut, 2, stride=2, padding=0, output_padding=0, bias=False)
        self.BR = BR(nOut)

    def forward(self, input):
        output = self.conv(input)
        output = self.BR(output)
        return output


class Basenet(nn.Module):
    def __init__(self, num_classes=1):
        super(Basenet, self).__init__()
        #downsample block 1
        self.downsample_1 = CBR(3,16,3,2)
        self.downsample_2 = CBR(16,64,3,2)
        self.regular_1 = nn.ModuleList()
        for i in range(5):
            self.regular_1.append(CBR(64,64,3,1))
        self.downsample_3 = CBR(64,128,3,2)
        self.regular_2 = nn.ModuleList()
        for i in range(8):
            self.regular_2.append(CBR(128, 128, 3, 1))
        self.Upsample_1 = upBR(128,64)
        self.regular_3 = nn.ModuleList()
        for i in range(2):
            self.regular_3.append(CBR(64, 64, 3, 1))
        self.Upsample_2 = upBR(64, 2*num_classes)
        self.regular_4 = nn.ModuleList()
        for i in range(2):
            self.regular_4.append(CBR(2*num_classes, 2*num_classes, 3, 1))
        self.Upsample_3 = upBR(2 * num_classes, num_classes)
        self.weights_init()

        self.vgg = VGG()
        self.load_vgg()
        self.spm = ScalePyramidModule()
        self.trans0 = nn.Conv2d(in_channels=896, out_channels=3, kernel_size=1, bias=False)

    def weights_init(self):
        for idx, m in enumerate(self.modules()):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):

        input = self.vgg(x)
        x = self.spm(*input)
        
        conv2_2, conv3_3, conv4_4, conv5_4 = x
        x = torch.cat([conv5_4, conv4_4], 1)

        x = self.trans0(F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True))

        downsample_1 = self.downsample_1(x)#/2
        downsample_2 = self.downsample_2(downsample_1)#/4
        x = downsample_2
        for regular_layer in self.regular_1:
            x = regular_layer(x)
        regular_1 = x
        downsample_3 = self.downsample_3(regular_1)#/8
        x = downsample_3
        for regular_layer in self.regular_2:
            x = regular_layer(x)
        regular_2 = x
        up_sample_1 = self.Upsample_1(regular_2)#/4
        x = up_sample_1
        for regular_layer in self.regular_3:
            x = regular_layer(x)
        regular_3 = x
        up_sample_2 = self.Upsample_2(regular_3)#/2
        x = up_sample_2
        for regular_layer in self.regular_4:
            x = regular_layer(x)
        regular_4 = x
        up_sample_3 = self.Upsample_3(regular_4)#/1
        return up_sample_3

    def load_vgg(self):
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth')
        old_name = [0,2,5,7,10,12,14,16,19,21,23,25,28,30,32,34]
        new_name = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_2', '3_3', '3_4', '4_1', '4_2', '4_3', '4_4', '5_1', '5_2', '5_3', '5_4']
        new_dict = {}
        for i in range(16):
            new_dict['conv' + new_name[i] + '.conv.weight'] = \
                state_dict['features.' + str(old_name[i]) + '.weight']
            new_dict['conv' + new_name[i] + '.conv.bias'] = \
                state_dict['features.' + str(old_name[i]) + '.bias']
        self.vgg.load_state_dict(new_dict, strict=False)


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        return self.relu(x)

class ASPP(nn.Module):
    def __init__(self, inplanes, BatchNorm):
        super(ASPP, self).__init__()
        dilations = [1, 12, 24, 36]

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(256*5, 256, 1, bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.relu(x)

        return self.dropout(x)
                
class ScalePyramidModule(nn.Module):
    def __init__(self):
        super(ScalePyramidModule, self).__init__()
        self.assp = ASPP(512, BatchNorm=None)
        self.can = ContextualModule(512, 512)
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, *input):
        conv2_2, conv3_3, conv4_4, conv5_4 = input 
        conv4_4 = self.can(conv4_4)
        ### Why don't you apply ASSP in higher resolution ??? ###
        conv5_4 = torch.cat([F.interpolate(self.assp(conv5_4), scale_factor=2, mode='bilinear', align_corners=True), 
                    self.reg_layer(F.interpolate(conv5_4, scale_factor=2, mode='bilinear', align_corners=True))], 1)
        
        return conv2_2, conv3_3, conv4_4, conv5_4

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_1 = BaseConv(3, 64, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv1_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        
        self.conv2_1 = BaseConv(64, 128, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        
        self.conv3_1 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv3_3 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv3_4 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        
        self.conv4_1 = BaseConv(256, 512, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv4_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv4_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv4_4 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        
        self.conv5_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv5_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv5_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv5_4 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)

    def forward(self, input):
        input = self.conv1_1(input)
        input = self.conv1_2(input)
        input = self.pool(input)
        input = self.conv2_1(input)
        conv2_2 = self.conv2_2(input)

        input = self.pool(conv2_2)
        input = self.conv3_1(input)
        input = self.conv3_2(input)
        input = self.conv3_3(input)
        conv3_4 = self.conv3_4(input)

        input = self.pool(conv3_4)
        input = self.conv4_1(input)
        input = self.conv4_2(input)
        input = self.conv4_3(input)
        conv4_4 = self.conv4_4(input)

        input = self.pool(conv4_4)
        input = self.conv5_1(input)
        input = self.conv5_2(input)
        input = self.conv5_3(input)
        conv5_4 = self.conv5_4(input)
        return conv2_2, conv3_4, conv4_4, conv5_4


class BackEnd(nn.Module):
    def __init__(self):
        super(BackEnd, self).__init__()
        
        self.conv1 = BaseConv(896, 256, 1, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        
        self.conv3 = BaseConv(896, 128, 1, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv4 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)

        self.conv5 = BaseConv(256, 64, 1, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv6 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv7 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)

    def forward(self, *input):
        conv2_2, conv3_3, conv4_4, conv5_4 = input
        
        input = torch.cat([conv5_4, conv4_4], 1)
        input = self.conv1(input)
        input = self.conv2(input)
        input = F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=True)
        
        input = torch.cat([input, conv3_3, F.interpolate(conv5_4, scale_factor=2, mode='bilinear', align_corners=True)], 1)
        input = self.conv3(input)
        input = self.conv4(input)
        input = F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=True)

        input = torch.cat([input, conv2_2], 1)
        input = self.conv5(input)
        input = self.conv6(input)
        input = self.conv7(input)

        return input


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)
        return input
'''
if __name__ == '__main__':

    x1 = torch.rand(1, 3, 544, 736)
    x2 = torch.rand(1, 3, 456, 600)
    x3 = torch.rand(1, 3, 272, 360)
    x4 = torch.rand(1, 3, 360, 480)
    model = Basenet()
    model.eval()
    for i in [x1, x2, x3, x4]:
        y = model(i)
        print(y.shape)
'''

if __name__ == '__main__':
    import time
    
    net = Basenet()
    x_image = Variable(torch.randn(1, 3, 224, 224))
    #y = net(x_image)

    # calculate model size
    print(f'    Param: {(sum(p.numel() for p in net.parameters())):,}')
    print('    Total params: %.2fMB' % (sum(p.numel() for p in net.parameters()) / (1024.0 * 1024) * 4))
    t1 = time.time()
    ##test for its speed on cpu
    for i in range(60):
        y0 = net(x_image)
    t2 = time.time()
    print('fps: ', 60 / (t2 - t1))
    print(y0.shape)



