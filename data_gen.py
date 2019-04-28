from blender.render_utils import OpenGLRenderer
import os
from config import cfg
import numpy as np
from base_utils import PoseTransformer, read_pose, read_pickle, save_pickle
import cv2
import random
from tqdm import tqdm

def fuse(img, mask, background):
    background = cv2.resize(background,(img.shape[1], img.shape[0]))
    silhouette = mask > 0
    background[silhouette] = img[silhouette]
    return background
    
use_background = True
r = OpenGLRenderer()
bg_imgs_path = os.path.join(cfg.DATA_DIR, 'bg_imgs.npy')
bg_imgs = np.load(bg_imgs_path)
for class_type in tqdm(cfg.linemod_cls_names):
    dir_path = os.path.join(cfg.LINEMOD_ORIG,'{}/data'.format(class_type))
    train_set = np.loadtxt(os.path.join(cfg.LINEMOD, '{}/training_range.txt'.format(class_type)),np.int32)
    output_path = os.path.join(cfg.DATA_DIR, 'renders/{}'.format(class_type))
    trans = PoseTransformer(class_type)
    os.makedirs(output_path, exist_ok=True)
    for idx in tqdm(train_set):
            rot_path = os.path.join(dir_path, 'rot{}.rot'.format(idx))
            tra_path = os.path.join(dir_path, 'tra{}.tra'.format(idx))
            pose = read_pose(rot_path, tra_path)
            pose = trans.orig_pose_to_blender_pose(pose)
            rot, tra = pose[:, :3], pose[:, 3]
            rgb, mask = r.render(class_type, pose, intrinsic_matrix=r.intrinsic_matrix['linemod'], render_type='all')
            rgb = rgb[:,:,[2,1,0]]  # opencv use bgr order instead of rgb
            if use_background:
                background = cv2.imread(bg_imgs[random.randint(0, bg_imgs.shape[0]-1)], 1)
            else:
                background = np.zeros_like(rgb)
            rgb = fuse(rgb, mask, background)
            retval = cv2.imwrite(os.path.join(output_path, '{}.jpg'.format(idx)), rgb)
            
