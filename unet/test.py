import os
import pylab

import unet.atest as atest

from unet.model import *
from unet.data import *
from unet.dcm_to_png import *

cwd = os.path.dirname(__file__)
model = unet(os.path.join(cwd, "model", 'unet_membrane.hdf5'))


def cnt(images):
    images = [image.astype(np.uint8) for image in images]
    testGene = testGenerator(images)
    results = model.predict_generator(testGene, len(images), verbose=1)
    masks = get_mask_img(results)

    cnt_list = atest.fusion(masks, images)
    return cnt_list
