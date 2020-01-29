# -*- coding: utf-8 -*-
"""
@author: ADIT CHANCHAL
"""


import argparse
import re
import os, glob, datetime
import numpy as np
from keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract
from keras.models import Model, load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
import data_generator as dg
import keras.backend as K

#%%
## Params
path = 'F:/year 3/btp/keras_code/data/train_images'
direc = 'F:/year 3/btp/keras_code/models'
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--train_data', default=path, type=str, help='path of train data')
parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--epoch', default=300, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_every', default=1, type=int, help='save model at every x epoches')
args = parser.parse_args()


save_dir = os.path.join(direc,args.model+'_'+'sigma'+str(args.sigma))

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

#%%
def DnCNN(depth,filters=64,image_channels=1, use_bnorm=True):
    layer_count = 0
    inpt = Input(shape=(None,None,image_channels),name = 'input'+str(layer_count))
    # 1st layer, Conv+relu
    layer_count += 1
    x = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),kernel_initializer='Orthogonal', padding='same',name = 'conv'+str(layer_count))(inpt)
    layer_count += 1
    x = Activation('relu',name = 'relu'+str(layer_count))(x)
    # depth-2 layers, Conv+BN+relu
    for i in range(depth-2):
        layer_count += 1
        x = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),kernel_initializer='Orthogonal', padding='same',use_bias = False,name = 'conv'+str(layer_count))(x)
        if use_bnorm:
            layer_count += 1
            #x = BatchNormalization(axis=3, momentum=0.1,epsilon=0.0001, name = 'bn'+str(layer_count))(x)
        x = BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'bn'+str(layer_count))(x)
        layer_count += 1
        x = Activation('relu',name = 'relu'+str(layer_count))(x)
    # last layer, Conv
    layer_count += 1
    x = Conv2D(filters=image_channels, kernel_size=(3,3), strides=(1,1), kernel_initializer='Orthogonal',padding='same',use_bias = False,name = 'conv'+str(layer_count))(x)
    layer_count += 1
    x = Subtract(name = 'subtract' + str(layer_count))([inpt, x])   # input - noise
    model = Model(inputs=inpt, outputs=x)

    return model

#%%

def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir,'model_*.hdf5'))  # get name list of all .hdf5 files
    #file_list = os.listdir(save_dir)
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).hdf5.*",file_)
            #print(result[0])
            epochs_exist.append(int(result[0]))
        initial_epoch=max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

def log(*args,**kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"),*args,**kwargs)

def lr_schedule(epoch):
    initial_lr = args.lr
    if epoch<=30:
        lr = initial_lr
    elif epoch<=60:
        lr = initial_lr/10
    elif epoch<=80:
        lr = initial_lr/20
    else:
        lr = initial_lr/20
    log('current learning rate is %2.8f' %lr)
    return lr
#%%
def train_datagen(epoch_iter=2000,epoch_num=5,batch_size=128,data_dir=args.train_data):
    print ( 'train_datagen' )
    while(True):
        n_count = 0
        if n_count == 0:
            print(n_count)
            xs = dg.datagenerator(data_dir)
            print ( xs.shape )

            #assert len(xs)%args.batch_size ==0, \

            print ( 'log done' )
            xs = xs/255.0
            xs = xs.astype('float32')

            indices = list(range(xs.shape[0]))
            n_count = 1
        print ( 'if done', range(epoch_num) )
        for _ in range(epoch_num):
            np.random.shuffle(indices)    # shuffle
            for i in range(0, len(indices), batch_size):
                batch_x = xs[indices[i:i+batch_size]]
                noise =  np.random.normal(0, args.sigma/255.0, batch_x.shape)    # noise
                #noise =  K.random_normal(ge_batch_y.shape, mean=0, stddev=args.sigma/255.0)
                batch_y = batch_x + noise
                yield batch_y, batch_x

# define loss
def sum_squared_error(y_true, y_pred):
    #return K.mean(K.square(y_pred - y_true), axis=-1)
    #return K.sum(K.square(y_pred - y_true), axis=-1)/2
    return K.sum(K.square(y_pred - y_true))/2

#%%

if __name__ == '__main__':
    # model selection
    model = DnCNN(depth=17,filters=64,image_channels=1,use_bnorm=True)
    model.summary()
    print ( 'DnCNN done' )
    #%%
    # load the last model in matconvnet style
    initial_epoch = findLastCheckpoint(save_dir=save_dir)
    if initial_epoch > 0:
        print('resuming by loading epoch %03d'%initial_epoch)
        model = load_model(os.path.join(save_dir,'model_%03d.hdf5'%initial_epoch), compile=False)

    # compile the model
    model.compile(optimizer=Adam(0.001), loss=sum_squared_error)
    print ( 'compile done' )
    # use call back functions
    checkpointer = ModelCheckpoint(os.path.join(save_dir,'model_{epoch:03d}.hdf5'),
                verbose=1, save_weights_only=False, period=args.save_every)

    #print ( 'checkpointer' )
    print ( checkpointer )
    print ( 'done checkpointer' )
    csv_logger = CSVLogger(os.path.join(save_dir,'log.csv'), append=True, separator=',')
    print ( csv_logger )
    print ( 'done csv_logger'  )
    lr_scheduler = LearningRateScheduler(lr_schedule)
    print ( lr_schedule )
    print ( 'done lr_schedule' )

    history = model.fit_generator(train_datagen(batch_size=args.batch_size),
                steps_per_epoch=2000, epochs=args.epoch, verbose=1, initial_epoch=initial_epoch,
                callbacks=[checkpointer,csv_logger,lr_scheduler])






