import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
#from misc.layer import convDU, convLR

class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output

class Nested_UNet_Efficient(nn.Module):

    def __init__(self, in_ch=3, out_ch=1,  pretrained=True, deep_supervision=False):
        super(Nested_UNet_Efficient, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32]

        self.deep_supervision = deep_supervision

        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
        self.conv5_0 = conv_block_nested(filters[4], filters[5], filters[5])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])
        self.conv4_1 = conv_block_nested(filters[4] + filters[5], filters[4], filters[4])

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2])
        self.conv3_2 = conv_block_nested(filters[3]*2 + filters[4], filters[3], filters[3])

        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1])
        self.conv2_3 = conv_block_nested(filters[2]*3 + filters[3], filters[2], filters[2])

        self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0])
        self.conv1_4 = conv_block_nested(filters[1]*4 + filters[2], filters[1], filters[1])
        
        self.conv0_5 = conv_block_nested(filters[0]*5 + filters[1], filters[0], filters[0])

        if self.deep_supervision:
            self.final1 = nn.Sequential(nn.Conv2d(filters[0], out_ch, kernel_size=1), self.activation)
            self.final2 = nn.Sequential(nn.Conv2d(filters[0], out_ch, kernel_size=1), self.activation)
            self.final3 = nn.Sequential(nn.Conv2d(filters[0], out_ch, kernel_size=1), self.activation)
            self.final4 = nn.Sequential(nn.Conv2d(filters[0], out_ch, kernel_size=1), self.activation)
        else:
            self.final = nn.Sequential(nn.Conv2d(filters[0], out_ch, kernel_size=1), self.activation)

        self.res = EfficientNet.from_pretrained('efficientnet-b7', advprop=True)
        #self.frontend = nn.Sequential(
        #   self.res._conv_stem, self.res._bn0, self.res._swish
        #)
        #self.dense = models.DenseNet()

        self.Expand1 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=1, bias=False)
        self.Expand2 = nn.Conv2d(in_channels=32, out_channels=256, kernel_size=1, bias=False)
        self.Expand3 = nn.Conv2d(in_channels=32, out_channels=512, kernel_size=1, bias=False)
        self.Expand4 = nn.Conv2d(in_channels=48, out_channels=1024, kernel_size=1, bias=False)
        self.Expand5 = nn.Conv2d(in_channels=48, out_channels=2048, kernel_size=1, bias=False)

    def forward(self, x):
        x_en = self.res._swish(self.res._bn0(self.res._conv_stem(x)))

        #layer 1 dan 2
        drop_connect_rate = self.res._global_params.drop_connect_rate
        if drop_connect_rate:
            drop_connect_rate *= float(0) / len(self.res._blocks) # scale drop connect_rate
        x_en = self.res._blocks[0](x_en, drop_connect_rate=drop_connect_rate)
        
        if drop_connect_rate:
            drop_connect_rate *= float(1) / len(self.res._blocks) # scale drop connect_rate
        x_en = self.res._blocks[1](x_en, drop_connect_rate=drop_connect_rate)

        #x0_0  = self.frontend(x)
        #print(x.size())
        x0_0  = self.conv0_0(x)
        #print(x0_0.size())
        #x1_0 = self.conv1_0(self.pool(x0_0))
        x1_0 = self.Expand1(x_en)
        x1_0 = F.interpolate(x1_0, scale_factor=2, mode='bilinear', align_corners=True)
        #print(x1_0.size())
        #print((F.interpolate(x1_0, scale_factor=2, mode='bilinear', align_corners=True)).size())
        x0_1 = self.conv0_1(torch.cat([x0_0, F.interpolate(x1_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))

        
        if drop_connect_rate:
            drop_connect_rate *= float(2) / len(self.res._blocks) # scale drop connect_rate
        x_en = self.res._blocks[2](x_en, drop_connect_rate=drop_connect_rate)

        #x2_0 = self.conv2_0(self.pool(x1_0))
        x2_0 = x_en
        x2_0 = self.Expand2(x_en)
        x2_0 = F.interpolate(x2_0, scale_factor=2, mode='bilinear', align_corners=True)
        #print(x2_0.size())
        x1_1 = self.conv1_1(torch.cat([x1_0, F.interpolate(x2_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, F.interpolate(x1_1, scale_factor=2, mode='bilinear', align_corners=True)], 1))

        #layer 3 dan 4
        
        if drop_connect_rate:
            drop_connect_rate *= float(3) / len(self.res._blocks) # scale drop connect_rate
        x_en = self.res._blocks[3](x_en, drop_connect_rate=drop_connect_rate)
        
        #x3_0 = self.conv3_0(self.pool(x2_0))
        x3_0 = x_en
        x3_0 = self.Expand3(x_en)
        x3_0 = F.interpolate(x3_0, scale_factor=2, mode='bilinear', align_corners=True)
        #print(x3_0.size())
        x2_1 = self.conv2_1(torch.cat([x2_0, F.interpolate(x3_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, F.interpolate(x2_1, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, F.interpolate(x1_2, scale_factor=2, mode='bilinear', align_corners=True)], 1))

        #layer 5 dan 6
        #if drop_connect_rate:
        #    drop_connect_rate *= float(3) / len(self.res._blocks) # scale drop connect_rate
        #x_en = self.res._blocks[3](x_en, drop_connect_rate=drop_connect_rate)

        if drop_connect_rate:
            drop_connect_rate *= float(4) / len(self.res._blocks) # scale drop connect_rate
        x_en = self.res._blocks[4](x_en, drop_connect_rate=drop_connect_rate)
        
        #x4_0 = self.conv4_0(self.pool(x3_0))
        x4_0 = x_en
        x4_0 = self.Expand4(x_en)
        #print(x4_0.size())
        x4_0 = F.interpolate(x4_0, scale_factor=2, mode='bilinear', align_corners=True)
        #print(x4_0.size())
        x3_1 = self.conv3_1(torch.cat([x3_0, F.interpolate(x4_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, F.interpolate(x3_1, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, F.interpolate(x2_2, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, F.interpolate(x1_3, scale_factor=2, mode='bilinear', align_corners=True)], 1))

        
        if drop_connect_rate:
            drop_connect_rate *= float(5) / len(self.res._blocks) # scale drop connect_rate
        x_en = self.res._blocks[5](x_en, drop_connect_rate=drop_connect_rate)

        
        #x5_0 = self.conv5_0(self.pool(x4_0))
        x5_0 = self.Expand5(x_en)
        #print(x5_0.size())
        x5_0 = F.interpolate(x5_0, scale_factor=2, mode='bilinear', align_corners=True)
        #print(x5_0.size())
        x4_1 = self.conv4_1(torch.cat([x4_0, F.interpolate(x5_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x3_2 = self.conv3_2(torch.cat([x3_0, x3_1, F.interpolate(x4_1, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x2_3 = self.conv2_3(torch.cat([x2_0, x2_1, x2_2, F.interpolate(x3_2, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x1_4 = self.conv1_4(torch.cat([x1_0, x1_1, x1_2, x1_3, F.interpolate(x2_3, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x0_5 = self.conv0_5(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, F.interpolate(x1_4, scale_factor=2, mode='bilinear', align_corners=True)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return (output1 + output2 + output3 + output4)/4

        else:
            output = self.final(x0_5)
            return output
            
            
if __name__ == '__main__':
    import time

    x = torch.rand((1, 3, 256, 256))
    lnet = Nested_UNet_Efficient(3, 1, 'test')
    # calculate model size
    print('    Total params: %.2fMB' % (sum(p.numel() for p in lnet.parameters()) / (1024.0 * 1024) * 4))
    t1 = time.time()
    ##test for its speed on cpu
    for i in range(60):
        y0 = lnet(x)
    t2 = time.time()
    print('fps: ', 60 / (t2 - t1))
    print(y0.shape)