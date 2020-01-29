# -*- coding: utf-8 -*-
"""
@author: ADIT CHANCHAL
"""

import argparse
import os
import time
import datetime
# import PIL.Image as Image
import numpy as np
from keras.models import load_model, model_from_json
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave
import matplotlib.pyplot as plt

#%%

path = 'F:/year 3/btp/keras_code/data/Test'
direc = 'F:/year 3/btp/keras_code/models'
# temp = direc + '\DnCNN_sigma25\ '


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default=path, type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default=['Set12'], type=list, help='name of test dataset')
    parser.add_argument('--sigma', default=25, type=int, help='noise level')
    # parser.add_argument('--model_dir', default=os.path.join(direc,'DnCNN_sigma25'), type=str, help='directory of the model')
    parser.add_argument('--model_dir', default='F:/year 3/btp/keras_code/models/DnCNN_sigma25', type=str, help='directory of the model')
    # parser.add_argument('--model_dir', default= temp, type=str, help='directory of the model')
    parser.add_argument('--model_name', default='model_001.hdf5', type=str, help='the model name')
    parser.add_argument('--result_dir', default='F:/year 3/btp/keras_code/results', type=str, help='directory of results')
    parser.add_argument('--save_result', default=0, type=int, help='save the denoised image, 1 or 0')
    return parser.parse_args()

#%%


def clean(pred):
    pred = pred + np.random.normal(0, 18 / 255.0, (256, 256))
    # plt.imshow(pred)
    return pred


#%%
def to_tensor(img):
    if img.ndim == 2:
        return img[np.newaxis, ..., np.newaxis]
    elif img.ndim == 3:
        return np.moveaxis(img, 2, 0)[..., np.newaxis]


def from_tensor(img):
    return np.squeeze(np.moveaxis(img[..., 0], 0, -1))


def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def save_result(result, path):
    path = path if path.find('.') != -1 else path + '.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        imsave(path, np.clip(result, 0, 1))


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()
#%%


if __name__ == '__main__':

    args = parse_args()


# os.path.exists(os.path.join(args.model_dir, args.model_name))
    if True:

        # json_file = open(os.path.join(args.model_dir,'model.json'), 'r')
        json_file = open('F:/year 3/btp/keras_code/models/DnCNN_sigma25/model.json', 'r')

        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        # model.load_weights(os.path.join(args.model_dir,'model.h5'))
        model.load_weights('F:/year 3/btp/keras_code/models/DnCNN_sigma25/model.h5')

        log('load trained model on Train400 dataset by kai')
    else:
        model = load_model(os.path.join(args.model_dir, args.model_name), compile=False)
        log('load trained model')
#%%
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
#%%
    for set_cur in args.set_names:

        if not os.path.exists(os.path.join(args.result_dir, set_cur)):
            os.mkdir(os.path.join(args.result_dir, set_cur))
        psnrs = []
        ssims = []

for sigma in range(10, 60, 5):

        psnr_sig = []
        ssim_sig = []
        for im in os.listdir(os.path.join(args.set_dir, set_cur)):
            if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):

                x = np.array(imread(os.path.join(args.set_dir, set_cur, im)), dtype=np.float32) / 255.0  # ORIGINAL CLEAN IMAGE
                np.random.seed(seed=0)  # for reproducibility
                y = x + np.random.normal(0, sigma / 255.0, x.shape)  # Add Gaussian noise without clipping
                y = y.astype(np.float32)
                y_ = to_tensor(y)
                start_time = time.time()
                x_ = model.predict(y_)  # PREDICTED CLEAN IMAGE

                elapsed_time = time.time() - start_time
                # print('%10s : %10s : %2.4f second'%(set_cur,im,elapsed_time))
                x_ = from_tensor(x_)

                psnr_x_ = compare_psnr(x, x_)  # PEAK SIGNAL TO NOISE RATION
                ssim_x_ = compare_ssim(x, x_)  # STRUCTURAL SIMILARITY INDEX
                psnr_sig.append(psnr_x_)
                ssim_sig.append(ssim_x_)

        psnr_avg = np.mean(psnr_sig)
        ssim_avg = np.mean(ssim_sig)
        psnrs.append(psnr_avg)
        ssims.append(ssim_avg)

        print(sigma)
        log('Datset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(set_cur, psnr_avg, ssim_avg))
#%%

 psnrs = [ 30.205032220101852,
 30.42110610320243,
 30.647215209182985,
 30.4500550381115,
 26.967157716215993,
 22.783733706330732,
 20.135313419120052,
 18.297705078580098,
 16.89791068740528,
 15.765029376604877 ]

ssims = [ 0.9116062001903283,
 0.9175246419655475,
 0.9256187374437116,
 0.9298127865308773,
 0.8193679343717856,
 0.6192985523756326,
 0.4860690980521274,
 0.40004073038640403,
 0.34011499634342335,
 0.29546414776505825 ]

noise = range(10, 60, 5)
plt.plot(noise, ssims)

