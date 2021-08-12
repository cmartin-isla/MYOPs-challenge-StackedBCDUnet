"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image,ImageOps
import util.util as util
import os
import torchvision.transforms as transforms



class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt

        label_paths, image_paths, instance_paths = self.get_paths(opt)
        print(label_paths, image_paths, instance_paths)

        util.natural_sort(label_paths)
        util.natural_sort(image_paths)
        if not opt.no_instance:
            util.natural_sort(instance_paths)

        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]
        instance_paths = instance_paths[:opt.max_dataset_size]

        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)

        self.label_paths = label_paths
        self.image_paths = image_paths
        self.instance_paths = instance_paths

        size = len(self.label_paths)
        self.dataset_size = size

    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        instance_paths = []
        assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths, instance_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def __getitem__(self, index):
        
        
        import matplotlib.pyplot as plt
        from util.util import tensor2im
        import numpy as np
        import random
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        data_transform_crop = transforms.Compose([
                transforms.CenterCrop(size = 256)

            ])
        
        
        
        data_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=20, translate=(0.05,0.05),scale=(0.8,1.2), shear=(0.2,0.2),resample=Image.NEAREST),
                transforms.RandomPerspective(distortion_scale =0.2,interpolation=Image.NEAREST),
                transforms.CenterCrop(size = 200)

            ])
        
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        label_2 = data_transform_crop(label)
        label = data_transform(label)
        
        label = label_2


        
        
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label)
         
        label_tensor = label_tensor* 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc
        


        # input image (real images)
        image_path = self.image_paths[index]
        assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)
        image = Image.open(image_path)
        image = image.convert('RGB')

        

        data_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=20, translate=(0.05,0.05),scale=(0.8,1.2), shear=(0.2,0.2),resample=Image.BICUBIC),
                transforms.RandomPerspective(distortion_scale =0.2,interpolation=Image.BICUBIC),
                transforms.CenterCrop(size = 200),
                transforms.ColorJitter(brightness = 0.5,contrast = 0.5),
                
            ])
        
        
        random.seed(seed) # apply this seed to img tranfsorms
        
        image_2 = data_transform_crop(image)
        image = data_transform(image)
        
        image = image_2
        
        """
        plt.imshow(np.vstack((
            np.hstack((np.asarray(label_2)*40,np.asarray(label)*40)),
            np.hstack((np.asarray(image_2)[...,0],np.asarray(image)[...,0]))
            )))
        plt.show()
        """
        
        
        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_path = self.instance_paths[index]
            instance = Image.open(instance_path)
            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size
