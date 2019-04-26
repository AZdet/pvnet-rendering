from blender.render_utils import OpenGLRenderer
import os
from config import cfg
import numpy as np
from base_utils import PoseTransformer, read_pose, read_pickle, save_pickle
# from scipy.misc import imshow, imsave, imread, imresize
from base_utils import PoseTransformer
import cv2

def fuse(img, mask, background):
    background = cv2.resize(background,(img.shape[1], img.shape[0]))
    silhouette = mask > 0
    background[silhouette] = img[silhouette]
    return background
    


class_type = 'cat'
dir_path = os.path.join(cfg.LINEMOD_ORIG,'{}/data'.format(class_type))
train_set = np.loadtxt(os.path.join(cfg.LINEMOD, '{}/training_range.txt'.format(class_type)),np.int32)

trans = PoseTransformer(class_type)

for idx in train_set:
        rot_path = os.path.join(dir_path, 'rot{}.rot'.format(idx))
        tra_path = os.path.join(dir_path, 'tra{}.tra'.format(idx))
        pose = read_pose(rot_path, tra_path) 
        pose = trans.orig_pose_to_blender_pose(pose)
        rot, tra = pose[:, :3], pose[:, 3]
        break
r = OpenGLRenderer()
rgb, mask = r.render(class_type, pose, intrinsic_matrix=r.intrinsic_matrix['linemod'], render_type='all')
rgb = rgb[:,:,[2,1,0]]  # opencv use bgr order instead of rgb
background = cv2.imread('Lena.png', 1)
#background = imread('Lena.png', 'RGB')
rgb = fuse(rgb, mask, background)
cv2.imwrite('RGB2.jpg', rgb)
#imsave("RGB2.jpg", rgb)