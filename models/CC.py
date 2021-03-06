import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class CrowdCounter(nn.Module):
    def __init__(self,gpus,model_name):
        super(CrowdCounter, self).__init__()        

        if model_name == 'AlexNet':
            from .SCC_Model.AlexNet import AlexNet as net        
        elif model_name == 'VGG':
            from .SCC_Model.VGG import VGG as net
        elif model_name == 'VGG_DECODER':
            from .SCC_Model.VGG_decoder import VGG_decoder as net
        elif model_name == 'MCNN':
            from .SCC_Model.MCNN import MCNN as net
        elif model_name == 'CSRNet':
            from .SCC_Model.CSRNet import CSRNet as net
        elif model_name == 'Res50':
            from .SCC_Model.Res50 import Res50 as net
        elif model_name == 'Res101':
            from .SCC_Model.Res101 import Res101 as net            
        elif model_name == 'Res101_SFCN':
            from .SCC_Model.Res101_SFCN import Res101_SFCN as net 
        elif model_name == 'UNet++':
            from .UNet.Nested_UNet_Efficient5 import Nested_UNet_Efficient5 as net
        elif model_name == 'UNet':
            from .UNet.Nested_UNet_Densenet5 import Nested_UNet_Densenet5 as net
        elif model_name == 'UNet3P':
            from .UNet.UNet3Plus import UNet3Plus as net
        elif model_name == 'UNet2D':
            from .UNet.Unet_2D import Unet_2D as net
        elif model_name == 'MnasNet':
            from .MnasNet import MnasNet as net
        elif model_name == 'BiSeNet':
            from .mobile.BiSeNet import BiSeNet as net
        elif model_name == 'Basenet':
            from .mobile.Basenet import Basenet as net
        elif model_name == 'ERFNet':
            from .mobile.ERFnet import Net as net
        elif model_name == 'EDANet':
            from .mobile.EDANet import EDANet as net
        elif model_name == 'ENet':
            from .mobile.Enet import ENet as net
        elif model_name == 'ESPNet':
            from .mobile.Espnet import ESPNet as net
        elif model_name == 'mobilenet':
            from .mobile.mobilenet import mbv2 as net

        if model_name == 'ERFNet':
            self.CCN = net(num_classes=1)
        elif model_name == 'ENet':
            self.CCN = net(num_classes=1)
        elif model_name == 'mobilenet':
            self.CCN = net(num_classes=1)
        else:
            self.CCN = net()

        if len(gpus)>1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN=self.CCN.cuda()
        self.loss_mse_fn = nn.MSELoss().cuda()
        #self.CCN=self.CCN.cpu()
        #self.loss_mse_fn = nn.MSELoss().cpu()
        
    @property
    def loss(self):
        return self.loss_mse
    
    def forward(self, img, gt_map):                               
        density_map = self.CCN(img)                          
        self.loss_mse= self.build_loss(density_map.squeeze(), gt_map.squeeze())    

        #loss1 = self.build_loss(density_map[0].squeeze(), gt_map.squeeze())
        #loss2 = self.build_loss(density_map[1].squeeze(), gt_map.squeeze())
        #loss3 = self.build_loss(density_map[2].squeeze(), gt_map.squeeze())
        #loss4 = self.build_loss(density_map[3].squeeze(), gt_map.squeeze())
        #loss5 = self.build_loss(density_map[4].squeeze(), gt_map.squeeze())
        #self.loss_mse = (loss1 + loss2 + loss3 + loss4 + loss5)/5         
        return density_map
    
    def build_loss(self, density_map, gt_data):
        loss_mse = self.loss_mse_fn(density_map, gt_data)  
        return loss_mse

    def test_forward(self, img):                               
        density_map = self.CCN(img)                    
        return density_map

