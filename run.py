from config import cfg
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--type', action='store', dest='type', type=str)
args = parser.parse_args()


def run_rendering():
    from blender.render_utils import Renderer, YCBRenderer, OpenGLRenderer
    # YCBRenderer.multi_thread_render()
    # renderer = YCBRenderer('037_scissors')
#     for cls_name in cfg.linemod_cls_names:
#         renderer=Renderer(cls_name)
#         renderer.run()
    renderer=OpenGLRenderer()
    renderer.run()


def run_fuse():
    from fuse.fuse import run
    run()


if __name__ == '__main__':
    globals()['run_' + args.type]()
