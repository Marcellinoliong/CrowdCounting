import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict

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

class Nested_UNet_Densenet(nn.Module):

    def __init__(self, in_ch=3, out_ch=1,  pretrained=True, deep_supervision=False):
        super(Nested_UNet_Densenet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.deep_supervision = deep_supervision

        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0])

        if self.deep_supervision:
            self.final1 = nn.Sequential(nn.Conv2d(filters[0], out_ch, kernel_size=1), self.activation)
            self.final2 = nn.Sequential(nn.Conv2d(filters[0], out_ch, kernel_size=1), self.activation)
            self.final3 = nn.Sequential(nn.Conv2d(filters[0], out_ch, kernel_size=1), self.activation)
            self.final4 = nn.Sequential(nn.Conv2d(filters[0], out_ch, kernel_size=1), self.activation)
        else:
            self.final = nn.Sequential(nn.Conv2d(filters[0], out_ch, kernel_size=1), self.activation)

        self.dense = models.densenet161(pretrained=True)

        #num_init_features = 96
        #growth_rate = 48
        #bn_size=4
        #drop_rate=0
        #self.frontend = nn.Sequential(OrderedDict([
        #    ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
        #    ('norm0', nn.BatchNorm2d(num_init_features)),
        #    ('relu0', nn.ReLU(inplace=True)),
        #    ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        #]))

        #num_features = num_init_features
        #self.block1 = _DenseBlock(num_layers=num_features, num_input_features=num_features,
        #                        bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        #num_features = num_features + self.dense.num_layers * growth_rate
        #self.trans1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)                       

        #self.block2 = _DenseBlock(num_layers=num_features, num_input_features=num_features,
        #                        bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        #num_features = num_features + self.dense.num_layers * growth_rate
        #self.trans2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)     

        #self.block3 = _DenseBlock(num_layers=num_features, num_input_features=num_features,
        #                        bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        #num_features = num_features + self.dense.num_layers * growth_rate
        #self.trans3 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)    

        #self.block4 = _DenseBlock(num_layers=num_features, num_input_features=num_features,
        #                        bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        #num_features = num_features + self.dense.num_layers * growth_rate
        #self.trans4 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)    

        #self.block5 = _DenseBlock(num_layers=num_features, num_input_features=num_features,
        #                        bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        #num_features = num_features + self.dense.num_layers * growth_rate
        #self.trans5 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)  

        #self.block6 = _DenseBlock(num_layers=num_features, num_input_features=num_features,
        #                        bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        #num_features = num_features + self.dense.num_layers * growth_rate
        #self.trans6 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)    

        #self.trans = nn.Conv2d(in_channels=2208, out_channels=64, kernel_size=1, bias=False)
        #self.avg = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        #x_dn = self.frontend(x)
        #x_dn = self.block1(x_dn)
        #x_dn = self.trans1(x_dn)
        #x_dn = self.block2(x_dn)
        #print(x_dn)
        #x_dn = self.trans2(x_dn)
        #print(x_dn)
        #x_dn = self.block3(x_dn)
        #x_dn = self.trans3(x_dn)
        #x_dn = self.block4(x_dn)
        #print(x_dn)
        #x_dn = self.trans4(x_dn)
        #print(x_dn)
        #x_dn = self.block5(x_dn)
        #x_dn = self.trans5(x_dn)
        #x_dn = self.block6(x_dn)
        #print(x_dn)
        #x_dn = self.trans6(x_dn)
        #print(x_dn)

        x_dn = self.dense.features(x)
        #print(x_dn.size())

        x_dn = self.trans(x_dn)
        #print(x_dn.size())
        x_dn = F.interpolate(x_dn, scale_factor=32, mode='bilinear', align_corners=True)
        #print(x_dn.size())

        #x0_0  = self.conv0_0(x)
        x0_0 = x_dn
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, F.interpolate(x1_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, F.interpolate(x2_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, F.interpolate(x1_1, scale_factor=2, mode='bilinear', align_corners=True)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, F.interpolate(x3_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, F.interpolate(x2_1, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, F.interpolate(x1_2, scale_factor=2, mode='bilinear', align_corners=True)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, F.interpolate(x4_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, F.interpolate(x3_1, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, F.interpolate(x2_2, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, F.interpolate(x1_3, scale_factor=2, mode='bilinear', align_corners=True)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return (output1 + output2 + output3 + output4)/4

        else:
            output = self.final(x0_4)
            return output
            
if __name__ == '__main__':
    import time

    x = torch.rand((1, 3, 256, 256))
    lnet = Nested_UNet_Densenet(3, 1, 'test')
    # calculate model size
    print('    Total params: %.2fMB' % (sum(p.numel() for p in lnet.parameters()) / (1024.0 * 1024) * 4))
    t1 = time.time()
    ##test for its speed on cpu
    for i in range(60):
        y0 = lnet(x)
    t2 = time.time()
    print('fps: ', 60 / (t2 - t1))
    print(y0.shape)