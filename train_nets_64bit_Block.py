# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 09:54:43 2020

@author: tarunyadav
"""


import os
import sys


import speck as sp
import simon as si

import numpy as np

from pickle import dump

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras import backend as K
from keras.regularizers import l2

bs = 5000;


def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr);
  return(res);

def make_checkpoint(datei):
  res = ModelCheckpoint(datei, monitor='val_acc', save_best_only = True);
  return(res);

#make residual tower of convolutional blocks
def make_resnet(num_blocks=1, num_filters=32, num_outputs=1, d1=1024, d2=1024, word_size=64, ks=3,depth=2, reg_param=0.0001, final_activation='sigmoid'):

  model_mlp = Sequential();
  dense1 = Dense(d1,activation="relu",input_shape=(num_blocks * word_size * 2,));
  model_mlp.add(dense1);

  for i in range(depth):
    dense2 = Dense(d2,activation="relu");
  out = Dense(num_outputs, activation=final_activation);
  model_mlp.add(out);
  return(model_mlp);

def train_speck_distinguisher(num_epochs, num_rounds=7, depth=1, neurons=1024, data_train=2**18, data_test=2**16,cipher="speck",difference=(0,0x0020)):
    #create the network
    net = make_resnet(depth=depth, reg_param=10**-5, d1=64, d2=neurons);
    net.summary();
    net.compile(optimizer='adam',loss='mse',metrics=['acc']);

    #generate training and validation data
    if (cipher=="speck"):
      wdir = './speck_nets/';
      X, Y = sp.make_train_data(data_train,num_rounds,diff=difference);
      X_eval, Y_eval = sp.make_train_data(data_test, num_rounds,diff=difference);
    elif (cipher=="simon"):
      wdir = './simon_nets/';
      X, Y = si.make_train_data(data_train,num_rounds,diff=difference);
      X_eval, Y_eval = si.make_train_data(data_test, num_rounds,diff=difference);
    print(difference);
    if not os.path.exists(wdir):
      os.makedirs(wdir)
    #set up model checkpoint
    check = make_checkpoint(wdir+'best_'+str(num_rounds)+'_depth_'+str(depth)+'_diff_'+str(difference)+'_data_train_'+str(data_train)+'_data_test_'+str(data_test)+"_epoch-{epoch:02d}_val_acc-{val_acc:.2f}" + '.h5');
    print("Trying to Find: " + wdir+'best_'+str(num_rounds)+'_depth_'+str(depth)+'_diff_'+str(difference)+'_data_train_'+str(data_train)+'_data_test_'+str(data_test)+'.h5' + " Data Used: " + str(data_train));
    #create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10,0.002, 0.0001));
    #train and evaluate
    h = net.fit(X,Y,epochs=num_epochs,batch_size=bs,validation_data=(X_eval, Y_eval), callbacks=[lr,check]);
    np.save(wdir+'h'+str(num_rounds)+'r_depth'+str(depth)+'.npy', h.history['val_acc']);
    np.save(wdir+'h'+str(num_rounds)+'r_depth'+str(depth)+'.npy', h.history['val_loss']);
    dump(h.history,open(wdir+'hist'+str(num_rounds)+'r_depth'+str(depth)+'.p','wb'));
    print("Best validation accuracy: ", np.max(h.history['val_acc']));
    return(net, h);
 