# -*- coding: utf-8 -*-
"""
@author: ADIT CHANCHAL
"""



import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

#%%

patch_size, stride = 40, 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128

#%%

def show(x,title=None,cbar=False,figsize=None):

    plt.figure(figsize=figsize)
    plt.imshow(x,interpolation='nearest',cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()

def data_aug(img, mode=0):

    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def gen_patches(file_name):

    # read image
    img = cv2.imread(file_name, 0)  # gray scale
    #img = cv2.resize( img, (20, 20) )
    h, w = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h*s),int(w*s)
        img_scaled = cv2.resize(img, (w_scaled,h_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled-patch_size+1, stride):
            for j in range(0, w_scaled-patch_size+1, stride):
                x = img_scaled[i:i+patch_size, j:j+patch_size]
                #patches.append(x)
                # data aug
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0,8))

                    if ( x_aug.shape[0]==40 and x_aug.shape[1]==40 ):
                        #print( x_aug.shape )
                        patches.append(x_aug)
  #  print ( len(patches) )
    return patches # (40 * 40) * m for 1 image

def datagenerator(data_dir,verbose=False):

    file_list = glob.glob(data_dir+'/*.png')  # get name list of all .png files
    final = np.array( (40*40,3645,2,1) )
    # initrialize
    data = []
    # generate patches
    for i in range(len(file_list)):
        patch = gen_patches(file_list[i])
        #t =np.array(patch)
        #print ( t.shape )
        data.append(patch)

        #print ( len(patch) )
        if verbose:
            print(str(i+1)+'/'+ str(len(file_list)) + ' is done ^_^')
    #return data
    #print(len( data[0][1][1] ) )
    data = np.array(data)
    #print ( data.shape[0] )
    #return data
    data = data.reshape((data[0][0].shape[0]*data[0][0].shape[1],data[0].shape[0],data.shape[0],1)) # redundant imension
    discard_n = len(data)-len(data)//batch_size*batch_size;
    data = np.delete(data,range(discard_n),axis = 0)
    print('^_^-training data finished-^_^')
    return data


#%%
if __name__ == '__main__':

    path = 'F:/year 3/btp/keras_code/data/train_images'
    data = datagenerator(data_dir=path)
    print ( data.shape )

#    print('Shape of result = ' + str(res.shape))
#    print('Saving data...')
#    if not os.path.exists(save_dir):
#            os.mkdir(save_dir)
#    np.save(save_dir+'clean_patches.npy', res)
    print('Done.')
