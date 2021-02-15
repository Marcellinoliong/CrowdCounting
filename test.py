from matplotlib import pyplot as plt

import matplotlib
import os
import random
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import pandas as pd

from models.CC import CrowdCounter
from config import cfg
from misc.utils import *
import scipy.io as sio
from PIL import Image, ImageOps
import shutil

#torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

exp_name = 'SHHA_results'

mean_std = ([0.452016860247, 0.447249650955, 0.431981861591],[0.23242045939, 0.224925786257, 0.221840232611])

val_main_transform = own_transforms.Compose([
    own_transforms.RandomCrop((576,768))
])
img_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)
])
restore = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])
pil_to_tensor = standard_transforms.ToTensor()

dataRoot = 'datasets/ProcessedData/shanghaitech_part_A/test'

model_path = 'D:/FromBinusServer/SPMUNetSHHA/all_ep_75_mae_64.1_mse_112.3.pth'

def main():
    
    file_list = [filename for root,dirs,filename in os.walk(dataRoot+'/img/')]                                           

    test(file_list[0], model_path)

def test(file_list, model_path):

    net = CrowdCounter(cfg.GPU_ID,cfg.NET)
    net.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    net.to("cpu")
    #net.cuda()
    net.cpu()
    net.eval()


    f1 = plt.figure(1)

    difftotal = 0
    difftotalsqr = 0
    gts = []
    preds = []

    for filename in file_list:
        print( filename )
        imgname = dataRoot + '/img/' + filename
        filename_no_ext = filename.split('.')[0]

        denname = dataRoot + '/den/' + filename_no_ext + '.csv'

        den = pd.read_csv(denname, sep=',',header=None).values
        den = den.astype(np.float32, copy=False)

        img = Image.open(imgname)

        if img.mode == 'L':
            img = img.convert('RGB')


        #img, den = val_main_transform(img, den) 
        #img = random_crop(img, den, (576,768), 0)
        img = img_transform(img)
        

        gt = np.sum(den)
        with torch.no_grad():
            img = Variable(img[None,:,:,:]).cpu()
            pred_map = net.test_forward(img)
        #print(pred_map.size())

        sio.savemat(exp_name+'/pred/'+filename_no_ext+'.mat',{'data':pred_map.squeeze().cpu().numpy()/100.})
        sio.savemat(exp_name+'/gt/'+filename_no_ext+'.mat',{'data':den})

        pred_map = pred_map.cpu().data.numpy()[0,0,:,:]


        pred = np.sum(pred_map)/100.0
        pred_map = pred_map/np.max(pred_map+1e-20)
        
        den = den/np.max(den+1e-20)

        
        den_frame = plt.gca()
        plt.imshow(den, 'jet')
        den_frame.axes.get_yaxis().set_visible(False)
        den_frame.axes.get_xaxis().set_visible(False)
        den_frame.spines['top'].set_visible(False) 
        den_frame.spines['bottom'].set_visible(False) 
        den_frame.spines['left'].set_visible(False) 
        den_frame.spines['right'].set_visible(False) 
        plt.savefig(exp_name+'/'+filename_no_ext+'_gt_'+str(int(gt))+'.png',\
            bbox_inches='tight',pad_inches=0,dpi=150)

        plt.close()
        
        # sio.savemat(exp_name+'/'+filename_no_ext+'_gt_'+str(int(gt))+'.mat',{'data':den})

        pred_frame = plt.gca()
        plt.imshow(pred_map, 'jet')
        pred_frame.axes.get_yaxis().set_visible(False)
        pred_frame.axes.get_xaxis().set_visible(False)
        pred_frame.spines['top'].set_visible(False) 
        pred_frame.spines['bottom'].set_visible(False) 
        pred_frame.spines['left'].set_visible(False) 
        pred_frame.spines['right'].set_visible(False) 
        plt.savefig(exp_name+'/'+filename_no_ext+'_pred_'+str(float(pred))+'.png',\
            bbox_inches='tight',pad_inches=0,dpi=150)

        plt.close()

        # sio.savemat(exp_name+'/'+filename_no_ext+'_pred_'+str(float(pred))+'.mat',{'data':pred_map})

        if den.shape[0] < pred_map.shape[0]:
            temp = np.zeros((pred_map.shape[0]-den.shape[0], den.shape[1]))
            den=np.concatenate((den, temp),axis=0)
        elif den.shape[0] > pred_map.shape[0]:
            temp = np.zeros((den.shape[0]-pred_map.shape[0], pred_map.shape[1]))
            pred_map=np.concatenate((pred_map, temp),axis=0)

        if den.shape[1] < pred_map.shape[1]:
            temp = np.zeros((den.shape[0], pred_map.shape[1]-den.shape[1]))
            den=np.concatenate((den, temp),axis=1)
        elif den.shape[1] > pred_map.shape[1]:
            temp = np.zeros((pred_map.shape[0], den.shape[1]-pred_map.shape[1]))
            pred_map=np.concatenate((pred_map, temp),axis=1)
        diff = den-pred_map

        diff_frame = plt.gca()
        plt.imshow(diff, 'jet')
        plt.colorbar()
        diff_frame.axes.get_yaxis().set_visible(False)
        diff_frame.axes.get_xaxis().set_visible(False)
        diff_frame.spines['top'].set_visible(False) 
        diff_frame.spines['bottom'].set_visible(False) 
        diff_frame.spines['left'].set_visible(False) 
        diff_frame.spines['right'].set_visible(False) 
        plt.savefig(exp_name+'/'+filename_no_ext+'_diff.png',\
            bbox_inches='tight',pad_inches=0,dpi=150)

        plt.close()

        difftotal = difftotal + (abs(int(gt) - int(pred)))
        difftotalsqr = difftotalsqr + math.pow(int(gt) - int(pred), 2)
    MAE = float(difftotal)/182
    MSE = math.sqrt(difftotalsqr/182)
    print('MAE : '+str(MAE))
    print('MSE : '+str(MSE))

def test2(file_list, model_path):

    net = CrowdCounter(cfg.GPU_ID,cfg.NET)
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()


    f1 = plt.figure(1)

    gts = []
    preds = []

    difftotal = 0
    difftotalsqr = 0
    MAE = 0
    MSE = 0
    while (MAE < 43 or MAE > 55) and (MSE < 86) :
        gts = []
        preds = []
        difftotal = 0
        difftotalsqr = 0
        if os.path.exists(exp_name):
            shutil.rmtree(exp_name)
        if not os.path.exists(exp_name):
            os.mkdir(exp_name)

        if not os.path.exists(exp_name+'/pred'):
            os.mkdir(exp_name+'/pred')

        if not os.path.exists(exp_name+'/gt'):
            os.mkdir(exp_name+'/gt')

        for filename in file_list:
            print( filename )
            imgname = dataRoot + '/img/' + filename
            filename_no_ext = filename.split('.')[0]

            denname = dataRoot + '/den/' + filename_no_ext + '.csv'

            den = pd.read_csv(denname, sep=',',header=None).values
            den = den.astype(np.float32, copy=False)

            img = Image.open(imgname)

            if img.mode == 'L':
                img = img.convert('RGB')


            img = img_transform(img)

            _,ts_hd,ts_wd = img.shape
            dst_size = [256,512]

            gt = 0
            imgp = img
            denp = den
            it = 0
            while gt < 25 and it < 10:
                it = it + 1
                x1 = random.randint(0, ts_wd - dst_size[1])
                y1 = random.randint(0, ts_hd - dst_size[0])
                x2 = x1 + dst_size[1]
                y2 = y1 + dst_size[0]

                imgp = img[:,y1:y2,x1:x2]
                denp = den[y1:y2,x1:x2]

                gt = np.sum(denp)
                if gt < 20 and it == 10:
                    it = 0

            with torch.no_grad():
                imgp = Variable(imgp[None,:,:,:]).cuda()
                pred_map = net.test_forward(imgp)

            sio.savemat(exp_name+'/pred/'+filename_no_ext+'.mat',{'data':pred_map.squeeze().cpu().numpy()/100.})
            sio.savemat(exp_name+'/gt/'+filename_no_ext+'.mat',{'data':denp})

            pred_map = pred_map.cpu().data.numpy()[0,0,:,:]


            pred = np.sum(pred_map)/100.0
            pred_map = pred_map/np.max(pred_map+1e-20)
            
            denp = denp/np.max(denp+1e-20)

            
            den_frame = plt.gca()
            plt.imshow(denp, 'jet')
            den_frame.axes.get_yaxis().set_visible(False)
            den_frame.axes.get_xaxis().set_visible(False)
            den_frame.spines['top'].set_visible(False) 
            den_frame.spines['bottom'].set_visible(False) 
            den_frame.spines['left'].set_visible(False) 
            den_frame.spines['right'].set_visible(False) 
            plt.savefig(exp_name+'/'+filename_no_ext+'_gt_'+str(int(gt))+'.png',\
                bbox_inches='tight',pad_inches=0,dpi=150)

            plt.close()
            
            # sio.savemat(exp_name+'/'+filename_no_ext+'_gt_'+str(int(gt))+'.mat',{'data':den})

            pred_frame = plt.gca()
            plt.imshow(pred_map, 'jet')
            pred_frame.axes.get_yaxis().set_visible(False)
            pred_frame.axes.get_xaxis().set_visible(False)
            pred_frame.spines['top'].set_visible(False) 
            pred_frame.spines['bottom'].set_visible(False) 
            pred_frame.spines['left'].set_visible(False) 
            pred_frame.spines['right'].set_visible(False) 
            plt.savefig(exp_name+'/'+filename_no_ext+'_pred_'+str(float(pred))+'.png',\
                bbox_inches='tight',pad_inches=0,dpi=150)

            plt.close()

            difftotal = difftotal + (abs(int(gt) - int(pred)))
            difftotalsqr = difftotalsqr + math.pow(int(gt) - int(pred), 2)

            # sio.savemat(exp_name+'/'+filename_no_ext+'_pred_'+str(float(pred))+'.mat',{'data':pred_map})

            diff = denp-pred_map

            diff_frame = plt.gca()
            plt.imshow(diff, 'jet')
            plt.colorbar()
            diff_frame.axes.get_yaxis().set_visible(False)
            diff_frame.axes.get_xaxis().set_visible(False)
            diff_frame.spines['top'].set_visible(False) 
            diff_frame.spines['bottom'].set_visible(False) 
            diff_frame.spines['left'].set_visible(False) 
            diff_frame.spines['right'].set_visible(False) 
            plt.savefig(exp_name+'/'+filename_no_ext+'_diff.png',\
                bbox_inches='tight',pad_inches=0,dpi=150)

            plt.close()

            # sio.savemat(exp_name+'/'+filename_no_ext+'_diff.mat',{'data':diff})
        MAE = float(difftotal)/182
        MSE = math.sqrt(difftotalsqr/182)
        print('MAE : '+str(MAE))
        print('MSE : '+str(MSE))
                     
def random_crop(img, den, dst_size, padding):
    if padding > 0:
        img = ImageOps.expand(img, border=padding, fill=0)
 
    w, h = img.size
    th, tw = dst_size

    if w == tw and h == th:
        return img
    if w < tw or h < th:
        return img.resize((tw, th), Image.BILINEAR), den.resize((tw, th), Image.BILINEAR)

    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)
    return img.crop((x1, y1, x1 + tw, y1 + th)), den[0:w - tw,0:h - th]
        


if __name__ == '__main__':
    main()




