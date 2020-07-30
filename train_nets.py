import os
import sys

# import comet_ml in the top of your file
#from comet_ml import Experiment
# Add the following code anywhere in your machine learning file
#experiment = Experiment(api_key="x8AcgAv6hHsfdtS6Zstmy91zX", project_name="differentialml", workspace="tarunyadav")

import speck as sp
import simon as si
import gift_64_opt as gi_64_opt
import gift_64_cofb as gi_64_cofb
import gimli_384
import numpy as np
#from os import urandom

from pickle import dump
from keras.models import load_model
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
def make_resnet(num_blocks=2, num_filters=32, num_outputs=1, d1=1024, d2=1024, word_size=16, ks=3,depth=2, reg_param=0.0001, final_activation='sigmoid'):
  #Input and preprocessing layers
  #inp = Input(shape=(num_blocks * word_size * 2,));
  #rs = Reshape((2 * num_blocks, word_size))(inp);
  #perm = Permute((2,1))(rs);
  #add a single residual layer that will expand the data to num_filters channels
  #this is a bit-sliced layer
  # conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm);
  # conv0 = BatchNormalization()(conv0);
  # conv0 = Activation('relu')(conv0);
  # #add residual blocks
  # shortcut = conv0;
  # for i in range(depth):
  #   conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut);
  #   conv1 = BatchNormalization()(conv1);
  #   conv1 = Activation('relu')(conv1);
  #   conv2 = Conv1D(num_filters, kernel_size=ks, padding='same',kernel_regularizer=l2(reg_param))(conv1);
  #   conv2 = BatchNormalization()(conv2);
  #   conv2 = Activation('relu')(conv2);
  #   shortcut = Add()([shortcut, conv2]);
  # #add prediction head
  # flat1 = Flatten()(shortcut);
  # dense1 = Dense(d1,kernel_regularizer=l2(reg_param))(flat1);
  # dense1 = BatchNormalization()(dense1);
  # dense1 = Activation('relu')(dense1);
  # dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1);
  # dense2 = BatchNormalization()(dense2);
  # dense2 = Activation('relu')(dense2);
  # out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2);
  # model = Model(inputs=inp, outputs=out);
  # return(model);
  #inp = Input(shape=(num_blocks * word_size * 2,));
  #rs = Reshape((2 * num_blocks, word_size),input_shape=(num_blocks * word_size * 2,));
  #perm = Permute((2,1));
  model_mlp = Sequential();
  #model_mlp.add(inp);
  #model_mlp.add(rs);
  #model_mlp.add(perm);
  dense1 = Dense(d1,activation="relu",input_shape=(num_blocks * word_size * 2,));
  #BN1 = BatchNormalization();
  #Act1 = Activation('relu');
  model_mlp.add(dense1);
  #model_mlp.add(BN1);
  #model_mlp.add(Act1);
  for i in range(depth):
    dense2 = Dense(d2,activation="relu");
    #BN2 = BatchNormalization();
    #Act2 = Activation('relu');
    model_mlp.add(dense2);
  #model_mlp.add(BN2);
  #model_mlp.add(Act2);
  #model_mlp.add(Flatten());
  #out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param));
  out = Dense(num_outputs, activation=final_activation);
  model_mlp.add(out);
  return(model_mlp);



def train_speck_distinguisher(num_epochs, num_rounds=7, depth=1, neurons=1024, data_train=2**18, data_test=2**16,cipher="speck",difference=(0,0x0020),start_round=1,pre_trained_model="fresh"):
    #create the network
    #generate training and validation data
    if (cipher=="speck"): 
      wdir = './speck_nets/';
      if (pre_trained_model=="fresh"):
          net = make_resnet(depth=depth, reg_param=10**-5, d1=32, d2=neurons,num_blocks=2,word_size=int(16/2));#word size decreased(8) because c0^c1
          net.compile(optimizer='adam',loss='mse',metrics=['acc']);
      else:
          net = load_model(pre_trained_model);
      net.summary();
      X, Y = sp.make_train_data(data_train,num_rounds,diff=difference,r_start=start_round);
      X_eval, Y_eval = sp.make_train_data(data_test, num_rounds,diff=difference,r_start=start_round);
    elif (cipher=="simon"):
      wdir = './simon_nets/';
      #net = make_resnet(depth=depth, reg_param=10**-5, d1=64, d2=neurons);
      #net.summary(); 
      #net.compile(optimizer='adam',loss='mse',metrics=['acc']);
      if (pre_trained_model=="fresh"):
          net = make_resnet(depth=depth, reg_param=10**-5, d1=32, d2=neurons,num_blocks=2,word_size=int(16/2));#word size decreased(8) because c0^c1
          net.compile(optimizer='adam',loss='mse',metrics=['acc']);
      else:
          net = load_model(pre_trained_model);
      net.summary();
      X, Y = si.make_train_data(data_train,num_rounds,diff=difference,r_start=start_round);
      X_eval, Y_eval = si.make_train_data(data_test, num_rounds,diff=difference,r_start=start_round);
    elif (cipher=="GIFT_64_ENCRYPT" or cipher=="GIFT_64_DECRYPT" ):
      wdir = './gift_64_encrypt_nets/';
      #net = make_resnet(depth=depth,d1=128,d2=neurons,num_blocks=4,word_size=16);
      #cipher text with XOR
      if (pre_trained_model=="fresh"): 
          net = make_resnet(depth=depth,d1=64,d2=neurons,num_blocks=4,word_size=int(16/2)); #word size decreased(8) because c0^c1
          net.compile(optimizer='adam',loss='mse',metrics=['acc']);
      else:
          net = load_model(pre_trained_model);
      net.summary();
      encrypt_data = 1;
      if(cipher=="GIFT_64_DECRYPT"):
          wdir = './gift_64_decrypt_nets/';
          encrypt_data = 0;
      
      X, Y = gi_64_opt.make_train_data(data_train,num_rounds,diff=difference,r_start=start_round,encrypt_data=encrypt_data);
      X_eval, Y_eval = gi_64_opt.make_train_data(data_test, num_rounds,diff=difference,r_start=start_round,encrypt_data=encrypt_data);
    elif (cipher=="GIFT_64_COFB"):
      wdir = './gift_64_cofb_nets/';
      net = make_resnet(depth=depth,d1=128,d2=neurons,num_blocks=4,word_size=16);
      net.summary();
      net.compile(optimizer='adam',loss='mse',metrics=['acc']);
      X, Y = gi_64_cofb.make_train_data(data_train,num_rounds,diff=difference);
      X_eval, Y_eval = gi_64_cofb.make_train_data(data_test, num_rounds,diff=difference);
    elif (cipher=="GIMLI_384"):
      wdir = './gimli_384_nets/';
      net = make_resnet(depth=depth,d1=384,d2=neurons,num_blocks=12,word_size=int(32/2)); #word size decreased(16) because c0^c1 as input
      net.summary();
      net.compile(optimizer='adam',loss='mse',metrics=['acc']);
      X, Y = gimli_384.make_train_data(data_train,num_rounds,diff=difference);
      X_eval, Y_eval = gimli_384.make_train_data(data_test, num_rounds,diff=difference);
    print(difference);
    if not os.path.exists(wdir):
      os.makedirs(wdir)
    #set up model checkpoint
    if (pre_trained_model=="fresh"): 
      check = make_checkpoint(wdir+'best_'+str(num_rounds)+'_start_'+str(start_round)+'_depth_'+str(depth)+'_diff_'+str(difference)+'_data_train_'+str(data_train)+'_data_test_'+str(data_test)+"_epoch-{epoch:02d}_val_acc-{val_acc:.2f}" + '.h5');
      print("Model will be stroed in File: " + wdir+'best_'+str(num_rounds)+'_start_'+str(start_round)+'_depth_'+str(depth)+'_diff_'+str(difference)+'_data_train_'+str(data_train)+'_data_test_'+str(data_test)+'.h5');
    else:
      check = make_checkpoint(pre_trained_model[:pre_trained_model.index("_epoch")]+"_epoch_imp-{epoch:02d}_val_acc_imp-{val_acc:.2f}" + '.h5'); 
      print("Model will be stroed in File: " + pre_trained_model[:pre_trained_model.index("_epoch")]+"_epoch_imp---_val_acc_imp----.h5");
    
    #create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10,0.002, 0.0001));
    #train and evaluate
    h = net.fit(X,Y,epochs=num_epochs,batch_size=bs,validation_data=(X_eval, Y_eval), callbacks=[lr,check]);
    np.save(wdir+'h'+str(num_rounds)+'r_depth'+str(depth)+'.npy', h.history['val_acc']);
    np.save(wdir+'h'+str(num_rounds)+'r_depth'+str(depth)+'.npy', h.history['val_loss']);
    dump(h.history,open(wdir+'hist'+str(num_rounds)+'r_depth'+str(depth)+'.p','wb'));
    print("Best validation accuracy: ", np.max(h.history['val_acc']));
    return(net, h);
 