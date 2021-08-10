"""
BOE Semantic segmentation Dataset Loader
"""
import os
import os.path as osp
import json
import numpy as np
from config import cfg
from runx.logx import logx
from datasets.base_loader import BaseLoader
from datasets.utils import make_dataset_folder
from datasets import uniform
ID_TO_NAME = {
0:"construction--flat--road",
1:"construction--flat--sidewalk",
2:"object--street-light",
3:"construction--structure--bridge",
4:"construction--structure--building",
5:"human",
6:"object--support--pole",
7:"marking--continuous--dashed",
8:"marking--continuous--solid",
9:"marking--discrete--crosswalk-zebra",
10:"nature--sand",
11:"nature--sky",
12:"nature--snow",
13:"nature--terrain",
14:"nature--vegetation",
15:"nature--water",
16:"object--vehicle--bicycle",
17:"object--vehicle--boat",
18:"object--vehicle--bus",
19:"object--vehicle--car",
20:"object--vehicle--caravan",
21:"object--vehicle--motorcycle",
22:"object--vehicle--on-rails",
23:"object--vehicle--truck",
24:"construction--flat--pedestrian-area",
25:"construction--structure--tunnel",
26:"nature--wasteland",
}

NAME_TO_ID = {}
for k,v in ID_TO_NAME.items():
    NAME_TO_ID[v] = k

NAME_TO_MAPILLARY_NAME= {
"construction--flat--road":"construction--flat--road",
"construction--flat--sidewalk":"construction--flat--sidewalk",
"object--street-light":"object--street-light",
"construction--structure--bridge":"construction--structure--bridge",
"construction--structure--building":"construction--structure--building",
"human":"human--person--individual",
"object--support--pole":"object--support--pole",
"marking--continuous--dashed":"marking--continuous--dashed",
"marking--continuous--solid":"marking--continuous--solid",
"marking--discrete--crosswalk-zebra":"marking--discrete--crosswalk-zebra",
"nature--sand":"nature--sand",
"nature--sky":"nature--sky",
"nature--snow":"nature--snow",
"nature--terrain":"nature--terrain",
"nature--vegetation":"nature--vegetation",
"nature--water":"nature--water",
"object--vehicle--bicycle":"object--vehicle--bicycle",
"object--vehicle--boat":"object--vehicle--boat",
"object--vehicle--bus":"object--vehicle--bus",
"object--vehicle--car":"object--vehicle--car",
"object--vehicle--caravan":"object--vehicle--caravan",
"object--vehicle--motorcycle":"object--vehicle--motorcycle",
"object--vehicle--on-rails":"object--vehicle--on-rails",
"object--vehicle--truck":"object--vehicle--truck",
"construction--flat--pedestrian-area":"construction--flat--pedestrian-area",
"construction--structure--tunnel":"construction--structure--tunnel",
"nature--wasteland":"void--ground",
}
class Loader(BaseLoader):
    num_classes = len(ID_TO_NAME)
    ignore_label = 255
    trainid_to_name = ID_TO_NAME
    color_mapping = []

    def __init__(self, mode, quality='semantic', joint_transform_list=None,
                 img_transform=None, label_transform=None, eval_folder=None):

        super(Loader, self).__init__(quality=quality,
                                     mode=mode,
                                     joint_transform_list=joint_transform_list,
                                     img_transform=img_transform,
                                     label_transform=label_transform)
        root = cfg.DATASET.MAPILLARY_DIR
        config_fn = os.path.join(root, 'config.json')
        self.fill_colormap_and_names(config_fn)

        ######################################################################
        # Assemble image lists
        ######################################################################
        if mode == 'folder':
            self.all_imgs = make_dataset_folder(eval_folder)
        else:
            splits = {'train': 'training',
                      'val': 'validation',
                      'test': 'testing'}
            split_name = splits[mode]
            img_ext = 'jpg'
            mask_ext = 'png'
            img_root = os.path.join(root, split_name, 'images')
            mask_root = os.path.join(root, split_name, 'boe_labels')
            self.all_imgs = self.find_images(img_root, mask_root, img_ext,
                                             mask_ext)
        logx.msg('all imgs {}'.format(len(self.all_imgs)))
        self.centroids = uniform.build_centroids(self.all_imgs,
                                                 self.num_classes,
                                                 self.train,
                                                 cv=cfg.DATASET.CV)
        self.build_epoch()

    def fill_colormap_and_names(self, config_fn):
        """
        Mapillary code for color map and class names

        Outputs
        -------
        self.trainid_to_name
        self.color_mapping
        """
        if not osp.exists(config_fn):
            print(f"Find file {config_fn} faild, use random color.")
            colormap = np.random.rand(255*3)*255
            colormap = colormap.astype(np.uint8)
            self.color_mapping = colormap
            return

        with open(config_fn) as config_file:
            config = json.load(config_file)
        config_labels = config['labels']

        mapillary_name2color = {}

        for i in range(0, len(config_labels)):
            label = config_labels[i]['name']
            color = config_labels[i]['color']
            mapillary_name2color[label] = color

        # calculate label color mapping
        colormap = []
        self.trainid_to_name = {}
        for i in range(0, len(ID_TO_NAME)):
            name = ID_TO_NAME[i]
            m_name = NAME_TO_MAPILLARY_NAME[name]
            color = mapillary_name2color[m_name]
            colormap = colormap + color
        colormap += [255,255,255]*(256-len(ID_TO_NAME))
        self.color_mapping = colormap