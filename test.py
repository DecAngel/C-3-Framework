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

from datasets.SHHB.setting import cfg_data

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

exp_name = './SHHB_results'
if not os.path.exists(exp_name):
    os.mkdir(exp_name)

if not os.path.exists(exp_name+'/pred'):
    os.mkdir(exp_name+'/pred')

if not os.path.exists(exp_name+'/gt'):
    os.mkdir(exp_name+'/gt')

mean_std = ([0.452016860247, 0.447249650955, 0.431981861591],[0.23242045939, 0.224925786257, 0.221840232611])
img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
restore = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])
gt_transform = standard_transforms.Compose([
    own_transforms.LabelNormalize(cfg_data.LOG_PARA)
])
pil_to_tensor = standard_transforms.ToTensor()

dataRoot = './shanghaitech_part_B/test'

model_path = 'all_ep_89_mae_8.5_mse_14.0.pth'

def main():
    
    file_list = [filename for root,dirs,filename in os.walk(dataRoot+'/img/')]                                           

    test(file_list[0], model_path)
   

def test(file_list, model_path):

    net = CrowdCounter(cfg.GPU_ID, cfg.NET)
    net.load_state_dict(torch.load(model_path), strict=False)
    net.cuda()
    net.eval()

    f1 = plt.figure(1)

    gts = []
    preds = []

    for filename in file_list:
        imgname = dataRoot + '/img/' + filename
        filename_no_ext = filename.split('.')[0]

        denname = dataRoot + '/den/' + filename_no_ext + '.csv'

        den = pd.read_csv(denname, sep=',',header=None).values
        den = den.astype(np.float32, copy=False)
        den = gt_transform(den)

        img = Image.open(imgname)

        if img.mode == 'L':
            img = img.convert('RGB')


        img = img_transform(img)
        img = torch.unsqueeze(img, 0)
        img = torch.cat([img]*6, 0)

        gt = float(den.sum())/cfg_data.LOG_PARA
        with torch.no_grad():
            img = Variable(img).cuda()
            pred_map = net.test_forward(img)
        pred = float(pred_map.sum().data/cfg_data.LOG_PARA)
        #sio.savemat(exp_name+'/pred/'+filename_no_ext+'.mat',{'data':pred_map.squeeze().cpu().numpy()/100.})
        #sio.savemat(exp_name+'/gt/'+filename_no_ext+'.mat',{'data':den})

        #pred_map = pred_map.cpu().data.numpy()[0,:,:,:]

        #pred = np.sum(pred_map)/100.0
        # pred_map = pred_map/np.max(pred_map+1e-20)

        # den = den/np.max(den+1e-20)

        '''
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

        # sio.savemat(exp_name+'/'+filename_no_ext+'_diff.mat',{'data':diff})
        '''

        gts.append(gt)
        preds.append(pred)
        print('---'+filename+'---')
        print('预测人数：'+str(pred))
        print('实际人数：'+str(gt))

    gts = np.array(gts)
    preds = np.array(preds)
    print(np.abs(gts-preds))


if __name__ == '__main__':
    main()




