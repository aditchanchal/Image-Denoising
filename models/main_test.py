# -*- coding: utf-8 -*-
"""
@author: ADIT CHANCHAL




import argparse
import os, time, datetime
import PIL.Image as Image
import numpy as np
from keras.models import load_model, model_from_json
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import cv2

#%%

path = 'F:/year 3/btp/keras_code/data/Test'
direc = 'F:/year 3/btp/keras_code/models'
#temp = direc + '\DnCNN_sigma25\ '

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default=path, type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default=['Set12'], type=list, help='name of test dataset')
    parser.add_argument('--sigma', default=25, type=int, help='noise level')
    #parser.add_argument('--model_dir', default=os.path.join(direc,'DnCNN_sigma25'), type=str, help='directory of the model')
    parser.add_argument('--model_dir', default='F:/year 3/btp/keras_code/models/DnCNN_sigma25', type=str, help='directory of the model')
    #parser.add_argument('--model_dir', default= temp, type=str, help='directory of the model')
    parser.add_argument('--model_name', default='model_001.hdf5', type=str, help='the model name')
    parser.add_argument('--result_dir', default='F:/year 3/btp/keras_code/results', type=str, help='directory of results')
    parser.add_argument('--save_result', default=0, type=int, help='save the denoised image, 1 or 0')
    return parser.parse_args()

#%%
def clean(pred):
    pred = pred + np.random.normal(0, 18/255.0,  (256, 256) )
    #plt.imshow(pred)
    return pred


#%%
def to_tensor(img):
    if img.ndim == 2:
        return img[np.newaxis,...,np.newaxis]
    elif img.ndim == 3:
        return np.moveaxis(img,2,0)[...,np.newaxis]

def from_tensor(img):
    return np.squeeze(np.moveaxis(img[...,0],0,-1))

def log(*args,**kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"),*args,**kwargs)

def save_result(result,path):
    path = path if path.find('.') != -1 else path+'.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt','.dlm'):
        np.savetxt(path,result,fmt='%2.4f')
    else:
        imsave(path,np.clip(result,0,1))


def show(x,title=None,cbar=False,figsize=None):

    plt.figure(figsize=figsize)
    plt.imshow(x,interpolation='nearest',cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()
#%%

if __name__ == '__main__':

    args = parse_args()


#os.path.exists(os.path.join(args.model_dir, args.model_name))
    if True:

        #json_file = open(os.path.join(args.model_dir,'model.json'), 'r')
        json_file = open('F:/year 3/btp/keras_code/models/DnCNN_sigma25/model.json', 'r')

        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        #model.load_weights(os.path.join(args.model_dir,'model.h5'))
        model.load_weights('F:/year 3/btp/keras_code/models/DnCNN_sigma25/model.h5')

       # log('load trained model on Train400 dataset')
    else:
        model = load_model(os.path.join(args.model_dir, args.model_name),compile=False)
        log('load trained model')
#%%
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
#%%
    for set_cur in args.set_names:

        if not os.path.exists(os.path.join(args.result_dir,set_cur)):
            os.mkdir(os.path.join(args.result_dir,set_cur))
        psnrs = []
        ssims = []


        for im in os.listdir(os.path.join(args.set_dir,set_cur)):
            if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
                #x = np.array(Image.open(os.path.join(args.set_dir,set_cur,im)), dtype='float32') / 255.0
                x = np.array(imread(os.path.join(args.set_dir,set_cur,im)), dtype=np.float32) / 255.0 # ORIGINAL CLEAN IMAGE
                np.random.seed(seed=0) # for reproducibility
                y = x + np.random.normal(0, args.sigma/255.0, x.shape) # Add Gaussian noise without clipping
                y = y.astype(np.float32)
                y_  = to_tensor(y)
                start_time = time.time()
                x_ = model.predict(y_) # PREDICTED CLEAN IMAGE

                elapsed_time = time.time() - start_time
                print('%10s : %10s : %2.4f second'%(set_cur,im,elapsed_time))
                x_=from_tensor(x_)
                #x_ = clean(x_)
                #x_ = x_.astype( x.dtype )



                psnr_x_ = compare_psnr(x, x_) # PEAK SIGNAL TO NOISE RATION
                ssim_x_ = compare_ssim(x, x_) # STRUCTURAL SIMILARITY INDEX


                #if args.save_result:
                #    name, ext = os.path.splitext(im)
                print(psnr_x_)
                print(ssim_x_)
                show(x_, im) # show the image
                #    save_result(x_,path=os.path.join(args.result_dir,set_cur,name+'_dncnn'+ext)) # save the denoised image
                psnrs.append(psnr_x_)
                ssims.append(ssim_x_)
                save_result(y,path='F:/year 3/btp/mid_term evaluation/input_noisy'+im)
                save_result(x_,path='F:/year 3/btp/mid_term evaluation/pred_clean'+im)
                save_result(x,path='F:/year 3/btp/mid_term evaluation/true_clean'+im)
                save_result(y-x_,path='F:/year 3/btp/mid_term evaluation/noise'+im)

        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)
        psnrs.append(psnr_avg)
        ssims.append(ssim_avg)

        if args.save_result:
            save_result(np.hstack((psnrs,ssims)),path=os.path.join(args.result_dir,set_cur,'results.txt'))

        log('Datset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(set_cur, psnr_avg, ssim_avg))
#%%

noisy = y_[0]
noisy = noisy[:, :, 0]
pred_clean = x_
true_clean = x
plt.imshow(noisy)
#%%
plt.imshow(pred_clean)
#%%
plt.imshow(true_clean)
#%%
noise = noisy - pred_clean
plt.figure(figsize=(50, 50))
figure, axarr = plt.subplots(2,2)
axarr[0,0].imshow(noisy, aspect = 'auto')
axarr[0,1].imshow(noise, aspect = 'auto')
axarr[1,0].imshow(pred_clean, aspect = 'auto')
axarr[1,1].imshow(true_clean, aspect = 'auto')


