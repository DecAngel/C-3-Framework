import torch
from torch.autograd import Variable
from config import cfg
from models.CC import CrowdCounter
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
from PIL import Image
import numpy as np
import pandas as pd
from datasets.SHHB.setting import cfg_data


mean_std = ([0.452016860247, 0.447249650955, 0.431981861591],[0.23242045939, 0.224925786257, 0.221840232611])
img_transform = standard_transforms.Compose([
    own_transforms.RandomCrop(cfg_data.TRAIN_SIZE),
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)
])
restore = standard_transforms.Compose([
    own_transforms.DeNormalize(*mean_std),
    standard_transforms.ToPILImage()
])
pil_to_tensor = standard_transforms.ToTensor()

pic_path = '3.jpg'
model_path = 'all_ep_89_mae_8.5_mse_14.0.pth'

net = CrowdCounter(cfg.GPU_ID, cfg.NET)
net.load_state_dict(torch.load(model_path), strict=False)
net.cuda()
net.eval()

img = Image.open(pic_path)
if img.mode == 'L':
    img = img.convert('RGB')
img = img_transform(img)

with torch.no_grad():
    img = Variable(img[None,:,:,:]).cuda()
    pred_map = net.test_forward(img)

pred_map = pred_map.cpu().data.numpy()[0,0,:,:]
pred = np.sum(pred_map)/100.0

print('预测人数：'+str(pred))
