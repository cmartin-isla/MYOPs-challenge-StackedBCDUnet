"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
from util.util import tensor2im
import numpy as np
opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# test_interface
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break
    
    import matplotlib.pyplot as plt
    import numpy as np
    #plt.imshow(np.squeeze(data_i['label'][0]))
    #plt.show()
    
    generated = model(data_i, mode='inference')

    img_path = data_i['path']
    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        visuals = OrderedDict([('input_label', data_i['label'][b]),
                               ('original_image', data_i['image'][b]),
                               ('synthesized_image', generated[b])])
        
        
        gen = np.squeeze(tensor2im(generated[0]))
        print(gen.shape)

        visualizer.save_images(webpage, visuals, img_path[b:b + 1])

webpage.save()
