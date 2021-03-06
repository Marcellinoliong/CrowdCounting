import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict
from .contextual_layer import ContextualModule
from torch.utils import model_zoo

class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, stride=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output

class conv_block_trans(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_trans, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=2, padding=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=2, padding=1, stride=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output

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
        #dilations = [1, 12, 24, 36]
        dilations = [1, 6, 12, 18]

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
        #conv5_4 = torch.cat([F.interpolate(self.assp(conv5_4), scale_factor=2, mode='bilinear', align_corners=True), 
        #           self.reg_layer(F.interpolate(conv5_4, scale_factor=2, mode='bilinear', align_corners=True))], 1)
        conv5_4 = self.assp(conv5_4)
        
        return conv2_2, conv3_3, conv4_4, conv5_4

class Nested_UNet_Densenet5(nn.Module):

    def __init__(self, in_ch=3, out_ch=1,  pretrained=True):
        super(Nested_UNet_Densenet5, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32]

        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
        #self.conv5_0 = conv_block_nested(filters[4], filters[5], filters[5])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])
        #self.conv4_1 = conv_block_nested(filters[4] + filters[5], filters[4], filters[4])

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2])
        #self.conv3_2 = conv_block_nested(filters[3]*2 + filters[4], filters[3], filters[3])

        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1])
        #self.conv2_3 = conv_block_nested(filters[2]*3 + filters[3], filters[2], filters[2])

        self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0])
        #self.conv1_4 = conv_block_nested(filters[1]*4 + filters[2], filters[1], filters[1])
        
        #self.conv0_5 = conv_block_nested(filters[0]*5 + filters[1], filters[0], filters[0])

        self.final = nn.Sequential(nn.Conv2d(filters[0], out_ch, kernel_size=1), self.activation)

        #self.dense = models.densenet161(pretrained=True) 

        self.vgg = VGG()
        self.load_vgg()
        self.spm = ScalePyramidModule()

        #self.trans = nn.Conv2d(in_channels=2208, out_channels=64, kernel_size=1, bias=False)
        self.trans0 = nn.Conv2d(in_channels=384, out_channels=64, kernel_size=1, bias=False)
        #self.trans1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, bias=False)
        #self.trans2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, bias=False)
        #self.trans3 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, bias=False)
        #self.trans4 = nn.Conv2d(in_channels=896, out_channels=1024, kernel_size=1, bias=False)
        #self.trans5 = nn.Conv2d(in_channels=2208, out_channels=2048, kernel_size=1, bias=False)

        #self.trans0 = conv_block_trans(128, 64, 64)
        #self.trans1 = conv_block_trans(256, 128, 128)
        #self.trans2 = conv_block_trans(512, 256, 256)
        #self.trans3 = conv_block_trans(1024, 512, 512)
        #self.trans4 = conv_block_trans(896, 1024, 1024)
        #self.trans5 = conv_block_trans(2208, 2048, 2048)
        #self.transspm = conv_block_trans(2208, 512, 512)

    def forward(self, x):
        #x_dn = self.dense.features(x)
        #print(x_dn.size())

        input = self.vgg(x)
        #x_dn = self.transspm(input)
        x_dn = self.spm(*input)

        conv2_2, conv3_3, conv4_4, conv5_4 = x_dn
        #x_out = torch.cat([conv5_4, conv4_4], 1)
        #print(conv5_4.size())
        #print(conv4_4.size())
        #print(conv3_3.size())
        #print(conv2_2.size())

        #x5_0 = self.trans5(x_dn)
        #x5_0 = x_dn
        #x4_0 = self.trans4(F.interpolate(x_out, scale_factor=8, mode='bilinear', align_corners=True))
        #x4_0 = x_out
        #x3_0 = self.trans3(x4_0)
        #x2_0 = self.trans2(x3_0)
        #x1_0 = self.trans1(x2_0)
        x0_0 = self.trans0(F.interpolate(conv5_4, scale_factor=8, mode='bilinear', align_corners=True))
        #x4_0 = self.trans4(F.interpolate(x5_0, scale_factor=2, mode='bilinear', align_corners=True))
        #x3_0 = self.trans3(F.interpolate(x4_0, scale_factor=2, mode='bilinear', align_corners=True))
        #x2_0 = self.trans2(F.interpolate(x3_0, scale_factor=2, mode='bilinear', align_corners=True))
        #x1_0 = self.trans1(F.interpolate(x2_0, scale_factor=2, mode='bilinear', align_corners=True))
        #x0_0 = self.trans0(F.interpolate(x1_0, scale_factor=2, mode='bilinear', align_corners=True))

        #x0_0  = self.conv0_0(x)
        #x_dn = self.trans(x_dn)
        #x0_0 = F.interpolate(x_dn, scale_factor=32, mode='bilinear', align_corners=True)

        #x1_0 = self.conv1_0(self.pool(x0_0))
        x1_0 = conv2_2
        x0_1 = self.conv0_1(torch.cat([x0_0, F.interpolate(x1_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))

        #x2_0 = self.conv2_0(self.pool(x1_0))
        x2_0 = conv3_3
        x1_1 = self.conv1_1(torch.cat([x1_0, F.interpolate(x2_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, F.interpolate(x1_1, scale_factor=2, mode='bilinear', align_corners=True)], 1))

        #x3_0 = self.conv3_0(self.pool(x2_0))
        x3_0 = conv4_4
        x2_1 = self.conv2_1(torch.cat([x2_0, F.interpolate(x3_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, F.interpolate(x2_1, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, F.interpolate(x1_2, scale_factor=2, mode='bilinear', align_corners=True)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, F.interpolate(x4_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, F.interpolate(x3_1, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, F.interpolate(x2_2, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, F.interpolate(x1_3, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        '''
        x5_0 = self.conv5_0(self.pool(x4_0))
        x4_1 = self.conv4_1(torch.cat([x4_0, F.interpolate(x5_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x3_2 = self.conv3_2(torch.cat([x3_0, x3_1, F.interpolate(x4_1, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x2_3 = self.conv2_3(torch.cat([x2_0, x2_1, x2_2, F.interpolate(x3_2, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x1_4 = self.conv1_4(torch.cat([x1_0, x1_1, x1_2, x1_3, F.interpolate(x2_3, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x0_5 = self.conv0_5(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, F.interpolate(x1_4, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        '''
        output = self.final(x0_4)
        return output

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

'''
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
'''


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
            
if __name__ == '__main__':
    import time

    x = torch.rand((1, 3, 576,768))
    lnet = Nested_UNet_Densenet5(3, 1, 'test')
    # calculate model size
    print(f'    Param: {(sum(p.numel() for p in lnet.parameters())):,}')
    print('    Total params: %.2fMB' % (sum(p.numel() for p in lnet.parameters()) / (1024.0 * 1024) * 4))
    t1 = time.time()
    ##test for its speed on cpu
    for i in range(60):
        y0 = lnet(x)
    t2 = time.time()
    print('fps: ', 60 / (t2 - t1))
    print(y0.shape)