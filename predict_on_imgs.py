from __future__ import absolute_import
from __future__ import division
import torchvision.transforms as standard_transforms
import os
import sys
import time
from PIL import Image
import torch
import glob
from config import assert_and_infer_cfg, update_epoch, cfg
from utils.misc import AverageMeter, prep_experiment, eval_metrics
from utils.misc import ImageDumper
from utils.trnval_utils import eval_minibatch, validate_topn
from loss.utils import get_loss
from loss.optimizer import get_optimizer, restore_opt, restore_net
from network.ocrnet import InferenceHRNet_Mscale
import numpy as np
import shutil

#wj
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# Enable CUDNN Benchmarking optimization
torch.backends.cudnn.benchmark = True
color_mapping = (np.random.rand(255*3)*255).astype(np.uint8)
def save_mask(image_array,save_path):
    new_mask = Image.fromarray(image_array.astype(np.uint8)).convert('P')
    new_mask.putpalette(color_mapping)
    new_mask.save(save_path)

def main():
    img_path = 'example_imgs'
    save_dir = 'logs/eval_boe/ocrnet.HRNet_Mscale_honored-boe_folderx'
    checkpoint_path = 'logs/train_boe/ocrnet.HRNet_Mscale_honored-boe/last_checkpoint_ep19.pth'

    cfg.MODEL.BNFUNC = torch.nn.BatchNorm2d
    cfg.DATASET.NUM_CLASSES = 27
    cfg.RESULT_DIR = save_dir
    cfg.MODEL.N_SCALES = [0.5,1.0,2.0]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_imgs = glob.glob(os.path.join(img_path,"*.jpg"))
    if len(all_imgs) == 0:
        all_imgs = glob.glob(os.path.join(img_path,"*.jpeg"))

    net = InferenceHRNet_Mscale(cfg.DATASET.NUM_CLASSES,None)
    net.eval()
    net.to(torch.device('cuda:0'))

    checkpoint = torch.load(checkpoint_path,
                            map_location=torch.device('cpu'))
    restore_net(net, checkpoint)


    transforms = standard_transforms.Compose([standard_transforms.ToTensor(),
    standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    for val_idx, img_path in enumerate(all_imgs):
        torch.cuda.empty_cache()
        img = Image.open(img_path).convert('RGB')
        img =  transforms(img)
        img = torch.unsqueeze(img,0)
        print(img.size())
        img = img.to(torch.device("cuda:0"))
        with torch.no_grad():
            mask = net(img)

        img_save_path = os.path.join(save_dir,os.path.basename(img_path))
        shutil.copy(img_path,img_save_path)        
        save_path = os.path.join(save_dir,os.path.splitext(os.path.basename(img_path))[0]+".png")
        save_mask(mask[0].cpu().numpy(),save_path)

if __name__ == '__main__':
    main()
