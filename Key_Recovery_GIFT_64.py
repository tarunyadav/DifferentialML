# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:03:00 2020

@author: tarunyadav
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 08:50:33 2020

@author: tarunyadav
"""

from keras.models import load_model
import numpy as np
import gift_64_opt as gi_64_opt
import sys
from os import urandom 
import os
import time
from tqdm.notebook import tqdm
import random
#from progressbar import ProgressBar
#pbar = ProgressBar()

# load model
model = load_model(sys.argv[1]);
file_name = os.path.basename(sys.argv[1]);
#model.summary();

model_reduced = load_model(sys.argv[2]);
file_nam_reducede = os.path.basename(sys.argv[2]);
#model_reduced.summary();

def convert_to_binary_64_block(arr,WORD_SIZE=16,NO_OF_WORDS=4):
  X = np.zeros((NO_OF_WORDS * WORD_SIZE,len(arr[0])),dtype=np.uint8);
  for i in range(NO_OF_WORDS * WORD_SIZE):
    index = i // WORD_SIZE;
    offset = WORD_SIZE - (i % WORD_SIZE) - 1;
    X[i] = (arr[index] >> offset) & 1;
  X = X.transpose();
  return(X);

n = int(sys.argv[3])**int(sys.argv[4]); #2**18
nr = int(sys.argv[5]); #9
r_start = int(sys.argv[6]); # 1
#r_mid = int(sys.argv[6]); # 5
key_bits_change = int(sys.argv[7]); # 6
key_bits_change_nr_1 = int(sys.argv[8]);
cutoff_prob = float(sys.argv[9]);
error_allowed = float(sys.argv[10]); #0.027s
#cutoff = int(sys.argv[9]); # 58
repeat_loop = int(sys.argv[11])+1;
range_key_allowed = int(sys.argv[12]);
#rand_range_counter = int(sys.argv[10]);

diff_default=(int(file_name.split('_')[7][1:-1].split(',')[0]),int(file_name.split('_')[7][1:-1].split(',')[1]),int(file_name.split('_')[7][1:-1].split(',')[2]),int(file_name.split('_')[7][1:-1].split(',')[3]))
diff = diff_default;
acc_arr = [];
for iterate in range(1,repeat_loop):
    print("\n############ START: SAMPLE NO. %d #################"%(iterate));
    orig_key = np.frombuffer(urandom(16),dtype=np.uint16).reshape(8,-1);
    keys = np.repeat(orig_key,n,axis=1);
    plain0_0 = np.frombuffer(urandom(2*n),dtype=np.uint16);
    plain0_1 = np.frombuffer(urandom(2*n),dtype=np.uint16);
    plain0_2 = np.frombuffer(urandom(2*n),dtype=np.uint16);
    plain0_3 = np.frombuffer(urandom(2*n),dtype=np.uint16);
    plain1_0 = plain0_0 ^ diff[0];
    plain1_1 = plain0_1 ^ diff[1]; 
    plain1_2 = plain0_2 ^ diff[2];
    plain1_3 = plain0_3 ^ diff[3];
    ks = gi_64_opt.expand_key(keys, (r_start-1) + nr);
    #print(len(ks));
    #print(len(ks[0]));
    #print(ks)
    #print(ks[nr-1][7])
    #print(ks[nr-1][:,0])
    #print(ks[nr-1,0])
    #sys.exit()
    orignal_key_str_nr_round = "".join([np.binary_repr(k_i) for k_i in ks[nr-1][:,0]]);
    print("Original Key for round no. %d is: %s \n"%(nr,orignal_key_str_nr_round));
    orignal_key_str_nr_1_round = "".join([np.binary_repr(k_i) for k_i in ks[nr-2][:,0]]);
    print("Original Key for round no. %d is: %s \n"%(nr-1,orignal_key_str_nr_1_round));
    #orignal_key_str = "".join([np.binary_repr(k_i).zfill(16) for k_i in orig_key[:,0]]);
    #print("Original Key is: %s \n"%(orignal_key_str));
    cdata0 = gi_64_opt.encrypt((plain0_0, plain0_1, plain0_2, plain0_3), ks,r_start);
    cdata1 = gi_64_opt.encrypt((plain1_0, plain1_1, plain1_2, plain1_3), ks,r_start);
    
    #rand_range = random.sample(range(0,2**key_bits_change),rand_range_counter);
    #print(np.uint32(ks[nr-1][6,0]) << 2);
    #rand_range.insert(0,(np.uint32(ks[nr-1][6,0]) << int(key_bits_change/2)) | np.uint32(ks[nr-1][7,0]) )
    Prob_arr = np.zeros((n,2**key_bits_change),dtype=np.float32);
    range_ = range(0,2**key_bits_change);
    range_nr_1 = range(0,2**key_bits_change_nr_1);
    #total = len(range_);
    #with tqdm(total=total, position=0, leave=True) as pbar:
      #for bits_key in tqdm(range_,position=0, leave=True):
    for bits_key in tqdm(range_):
      #for bits_key in tqdm(rand_range):
          # to change round key only
          trial_keys = np.copy(ks); 
          trial_keys[nr-1][6] = 0;
          trial_keys[nr-1][7] = 0;
          #print(trial_keys[nr-1][7][0])
          trial_keys[nr-1][7] = ((trial_keys[nr-1][7]>>key_bits_change)<<key_bits_change) | bits_key;
          #trial_keys[nr-1][7] = ((trial_keys[nr-1][7]>>int(key_bits_change/2))<<int(key_bits_change/2)) | (bits_key & 0xffff);
          #trial_keys[nr-1][6] = ((trial_keys[nr-1][6]>>int(key_bits_change/2))<<int(key_bits_change/2)) | ((bits_key>>int(key_bits_change/2)) & 0xffff) ;
          #trial_key_str = "".join([np.binary_repr(k_i) for k_i in trial_keys[nr-1][:,0]]);
          #key_change = np.copy(keys); 
          #key_change[7] = ((key_change[7]>>key_bits_change)<<key_bits_change) | bits_key;
          #trial_keys = gi_64_opt.expand_key(key_change, (r_start-1) + nr);
          #trial_key_str = "".join([np.binary_repr(k_i) for k_i in key_change[:,0]]);
          #print("9th round key");
          #print("".join([np.binary_repr(k_i) for k_i in trial_keys[nr-1][:,0]]));
          cdata0_dec = gi_64_opt.decrypt(cdata0, trial_keys, nr,r_end = nr-1);
          cdata1_dec = gi_64_opt.decrypt(cdata1, trial_keys, nr,r_end = nr-1);
          X = convert_to_binary_64_block(np.array(cdata0_dec^cdata1_dec),16,4);
          Y_Prob = model.predict_proba(X,batch_size=5000);
          #print(Y_Prob[0]);
          Y_Prob[Y_Prob == 1] = 0.99999999;
          Y_Prob = Y_Prob/(1- Y_Prob); Y_Prob = np.log2(Y_Prob);
          #print(np.shape(Y_Prob));
          #print(np.shape(Prob_arr));
          Prob_arr[:,bits_key] = Y_Prob[:,0];
          #pbar.update();
          #Y_Prob_good =(np.max(Y_Prob)-error_allowed);
          #key_prediction_array.append(len(np.where(Y_Prob >= Y_Prob_good)[0]));
          
          #print("\n############ STARTING TRIAL NO. %d #################"%(bits_key));
          #print("For Trial Key ::%s::\n"%(trial_key_str));
          #print("Predicted(Probability >= : %s) No. of desired output diff %s after %d rounds is : %s"%(str(Y_Prob_good),str(diff_default),r_mid,len(np.where(Y_Prob >= Y_Prob_good)[0])));
          #print("############ END TRIAL NO. %d #################\n"%(bits_key));
    #change in original key
    #orignal_key_str_last_bits = orignal_key_str[-1*key_bits_change:];
    #print("\nOrignal Key last %d bits : %s (%d)"%(key_bits_change,orignal_key_str_last_bits,int(orignal_key_str_last_bits,2)));
    #print("Correct Key Guesses (Using Cutoff %d): "%(cutoff));
    #print([((np.binary_repr(i)).zfill(key_bits_change),i,key_prediction_array[i]) for i in range(len(key_prediction_array)) if key_prediction_array[i] > cutoff]);
    #print("Guess for Correct Key: ");
    #print(((np.binary_repr(int(orignal_key_str_last_bits,2))).zfill(key_bits_change),int(orignal_key_str_last_bits,2),key_prediction_array[int(orignal_key_str_last_bits,2)]))
    #print("All Key Guesses:")
    #print([((np.binary_repr(i)).zfill(key_bits_change),i,key_prediction_array[i]) for i in range(0,len(key_prediction_array))]);
    
    # change in round key
    # orignal_key_str_nr_round_last_bits = orignal_key_str_nr_round[-1*key_bits_change:];
    # print("\nOrignal Key last %d bits : %s (%d)"%(key_bits_change,orignal_key_str_nr_round_last_bits,int(orignal_key_str_nr_round_last_bits,2)));
    # print("Correct Key Guesses (Using Cutoff %d): "%(cutoff));
    # print([((np.binary_repr(i)).zfill(key_bits_change),i,key_prediction_array[i]) for i in range(len(key_prediction_array)) if key_prediction_array[i] > cutoff]);
    # #print([((np.binary_repr(rand_range[i])).zfill(key_bits_change),rand_range[i],key_prediction_array[i]) for i in range(len(key_prediction_array)) if key_prediction_array[i] > cutoff]);
    # print("Guess for Correct Key: ");
    # print(((np.binary_repr(int(orignal_key_str_nr_round_last_bits,2))).zfill(key_bits_change),int(orignal_key_str_nr_round_last_bits,2),key_prediction_array[int(orignal_key_str_nr_round_last_bits,2)]))
    # #print(((np.binary_repr(int(orignal_key_str_nr_round_last_bits,2))).zfill(key_bits_change),int(orignal_key_str_nr_round_last_bits,2),key_prediction_array[0]))
    # print("All Key Guesses:")
    # print([((np.binary_repr(i)).zfill(key_bits_change),i,key_prediction_array[i]) for i in range(0,len(key_prediction_array))]);
    # #print([((np.binary_repr(rand_range[i])).zfill(key_bits_change),rand_range[i],key_prediction_array[i]) for i in range(0,len(key_prediction_array))]);
    # print("############ END: SAMPLE NO. %d #################\n"%(iterate));
    #while len(tqdm._instances) > 0:
    tqdm._instances.clear()
    # cipher filtreing
    Prob_sum = np.sum(Prob_arr,axis=1);
    # print(np.max(Prob_sum));
    # print(len(Prob_sum))
    # #print(Prob_sum[Prob_sum>1]);
    # print(len(np.where(Prob_sum > 10 )[0]));
    #key_prediction_array = [];
    key_prediction_array = np.zeros((2**key_bits_change,2**key_bits_change_nr_1),dtype=np.uint32)
    cdata0_good = cdata0[:,np.where(Prob_sum > cutoff_prob)[0]];
    cdata1_good = cdata1[:,np.where(Prob_sum > cutoff_prob)[0]];
    print("No. of Good Cihper Pairs are %d"%(len(np.where(Prob_sum > cutoff_prob)[0])));
    #range_ = range(0,2**key_bits_change);
    #total = len(range_);
    #with tqdm(total=total, position=0, leave=True) as pbar:
      #for bits_key in tqdm(range_,position=0, leave=True):
    for bits_key in tqdm(range_):
          trial_keys = np.take(ks,np.where(Prob_sum > cutoff_prob)[0],axis=2);
          trial_keys[nr-1][6] = 0; 
          trial_keys[nr-1][7] = 0; 
          trial_keys[nr-1][7] = ((trial_keys[nr-1][7]>>key_bits_change)<<key_bits_change) | bits_key;
          cdata0_dec = gi_64_opt.decrypt(cdata0_good, trial_keys, nr,r_end = nr-1);
          cdata1_dec = gi_64_opt.decrypt(cdata1_good, trial_keys, nr,r_end = nr-1);
          
          for bits_key_reduced in range_nr_1:
               trial_keys[nr-2][6] = 0;
               trial_keys[nr-2][7] = 0;
               trial_keys[nr-2][7] = ((trial_keys[nr-2][7]>>key_bits_change_nr_1)<<key_bits_change_nr_1) | bits_key_reduced;
               cdata0_dec_reduced = gi_64_opt.decrypt(cdata0_dec, trial_keys, nr-1,r_end = nr-2);
               cdata1_dec_reduced = gi_64_opt.decrypt(cdata1_dec, trial_keys, nr-1,r_end = nr-2);
               X = convert_to_binary_64_block(np.array(cdata0_dec_reduced^cdata1_dec_reduced),16,4);
               Y_Prob = model_reduced.predict_proba(X,batch_size=5000);
               Y_Prob_good =(np.max(Y_Prob)-error_allowed);
               #key_prediction_array.append(len(np.where(Y_Prob >= Y_Prob_good)[0]));
               key_prediction_array[bits_key][bits_key_reduced] = len(np.where(Y_Prob >= Y_Prob_good)[0]);
               
          #pbar.update();
    #while len(tqdm._instances) > 0:
    tqdm._instances.clear()
    # #change in round key
    # orignal_key_str_nr_round_last_bits = orignal_key_str_nr_round[-1*key_bits_change:];
    # print("\nOrignal Key last %d bits : %s (%d)"%(key_bits_change,orignal_key_str_nr_round_last_bits,int(orignal_key_str_nr_round_last_bits,2)));
    # print("Correct Key Guesses (Using Cutoff %d): "%(cutoff));
    # correct_key_arr = [((np.binary_repr(i)).zfill(key_bits_change),i,key_prediction_array[i]) for i in range(len(key_prediction_array)) if key_prediction_array[i] > cutoff];
    # correct_key_arr.sort(reverse=True,key=lambda x:x[2]);
    # print(correct_key_arr);
    # #print([((np.binary_repr(rand_range[i])).zfill(key_bits_change),rand_range[i],key_prediction_array[i]) for i in range(len(key_prediction_array)) if key_prediction_array[i] > cutoff]);
    # print("Guess for Correct Key: ");
    # print(((np.binary_repr(int(orignal_key_str_nr_round_last_bits,2))).zfill(key_bits_change),int(orignal_key_str_nr_round_last_bits,2),key_prediction_array[int(orignal_key_str_nr_round_last_bits,2)]))
    # #print(((np.binary_repr(int(orignal_key_str_nr_round_last_bits,2))).zfill(key_bits_change),int(orignal_key_str_nr_round_last_bits,2),key_prediction_array[0]))
    # print("All Key Guesses:")
    # print([((np.binary_repr(i)).zfill(key_bits_change),i,key_prediction_array[i]) for i in range(0,len(key_prediction_array))]);
    # #print([((np.binary_repr(rand_range[i])).zfill(key_bits_change),rand_range[i],key_prediction_array[i]) for i in range(0,len(key_prediction_array))]);
    # print("############ END: SAMPLE NO. %d #################\n"%(iterate));
        #change in round key
    orignal_key_str_nr_round_last_bits = orignal_key_str_nr_round[-1*key_bits_change:];
    print("\nOrignal Key for round %d last %d bits : %s (%d)"%(nr,key_bits_change,orignal_key_str_nr_round_last_bits,int(orignal_key_str_nr_round_last_bits,2)));
    orignal_key_str_nr_1_round_last_bits = orignal_key_str_nr_1_round[-1*key_bits_change_nr_1:];
    print("Orignal Key for round %d last %d bits : %s (%d)"%(nr-1,key_bits_change_nr_1,orignal_key_str_nr_1_round_last_bits,int(orignal_key_str_nr_1_round_last_bits,2)));
    final_key_prediction = [(np.binary_repr(i).zfill(key_bits_change),np.binary_repr(j).zfill(key_bits_change_nr_1),key_prediction_array[i][j]) for (i ,j) in np.ndindex(np.shape(key_prediction_array))];
    final_key_prediction.sort(reverse=True,key=lambda x:x[2]);
    prediction_arr = np.array(final_key_prediction);
    occurance_nr_key = np.where(prediction_arr[:,0]==orignal_key_str_nr_round_last_bits)[0];
    occurance_nr_1_key = np.where(prediction_arr[:,1]==orignal_key_str_nr_1_round_last_bits)[0]; 
    print("First Occurance of Key(%s) for round %d is at: %d"%(orignal_key_str_nr_round_last_bits,nr, occurance_nr_key[0]))
    print("First Occurance of Key(%s) for round %d is at: %d"%(orignal_key_str_nr_1_round_last_bits,nr-1,occurance_nr_1_key[0]))
    print("First Occurance of (Key1(%s),Key2(%s)) is at: %d"%(orignal_key_str_nr_round_last_bits,orignal_key_str_nr_1_round_last_bits,np.intersect1d(occurance_nr_key,occurance_nr_1_key)[0] ))
    print(final_key_prediction[0:min(100,len(final_key_prediction))])
    #print(final_key_prediction);
    acc_arr.append((occurance_nr_key[0],occurance_nr_1_key[0],np.intersect1d(occurance_nr_key,occurance_nr_1_key)[0]))
    print("############ END: SAMPLE NO. %d #################\n"%(iterate));
np_acc_arr = np.array(acc_arr);
print("Total No. of Samples are %d"%(repeat_loop-1));
print("No. of Key for round %d in range %d is : %d"%(nr,range_key_allowed, len(np.where(np_acc_arr[:,0] <= range_key_allowed)[0])));
print("No. of Key for round %d in range %d is : %d"%(nr-1,range_key_allowed, len(np.where(np_acc_arr[:,1] <= range_key_allowed)[0])));
print(acc_arr);