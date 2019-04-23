from blender.render_utils import OpenGLRenderer
import os
from config import cfg
import numpy as np
from base_utils import PoseTransformer, read_pose, read_pickle, save_pickle
from scipy.misc import imshow, imsave
class_type = 'cat'
dir_path = os.path.join(cfg.LINEMOD_ORIG,'{}/data'.format(class_type))
train_set = np.loadtxt(os.path.join(cfg.LINEMOD, '{}/training_range.txt'.format(class_type)),np.int32)
for idx in train_set:
        rot_path = os.path.join(dir_path, 'rot{}.rot'.format(idx))
        tra_path = os.path.join(dir_path, 'tra{}.tra'.format(idx))
        pose = read_pose(rot_path, tra_path) 
        rot, tra = pose[:, :3], pose[:, 3]
        break
r = OpenGLRenderer()
rgb = r.render(class_type, pose, intrinsic_matrix=r.intrinsic_matrix['linemod'], render_type='rgb')
imshow(rgb) 
#imsave(rgb, "RGB")