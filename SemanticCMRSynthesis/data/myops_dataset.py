"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os.path
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset



class MyopsDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='none')
        if is_train:
            parser.set_defaults(load_size=256)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=160)
        parser.set_defaults(contain_dontcare_label=False)
        parser.set_defaults(cache_filelist_read=True)
        parser.set_defaults(cache_filelist_write=True)
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
            
        phase = 'val' if opt.phase == 'test' else opt.phase

        label_dir = os.path.join(root, '%s_label' % phase)
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)

        image_dir = os.path.join(root, '%s_img' % phase)
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)


        if not opt.no_instance:
            instance_dir = os.path.join(root, '%s_inst' % phase)
            instance_paths = make_dataset(instance_dir, recursive=False, read_cache=True)

        else:
            instance_paths = []

        return label_paths, image_paths, instance_paths
