# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 09:23:15 2020

@author: tarunyadav
"""
import numpy as np
from os import urandom
# int(np.binary_repr(3).zfill(4)[::-1],2)
GIFT_S = [1,10, 4,12, 6,15, 3, 9, 2,13,11, 7, 5, 0, 8,14];
#GIFT_S = [8,5, 2,3, 6,15, 12, 9, 4,11,13, 14, 10, 0, 1,7]; #buggy s-box
GIFT_S_inv = [13, 0, 8, 6, 2,12, 4,11,14, 7, 1,10, 3, 9,15, 5];
GIFT_P = [0, 17, 34, 51, 48,  1, 18, 35, 32, 49,  2, 19, 16, 33, 50,  3,
  4, 21, 38, 55, 52,  5, 22, 39, 36, 53,  6, 23, 20, 37, 54,  7,
  8, 25, 42, 59, 56,  9, 26, 43, 40, 57, 10, 27, 24, 41, 58, 11,
 12, 29, 46, 63, 60, 13, 30, 47, 44, 61, 14, 31, 28, 45, 62, 15 ]
GIFT_P_inv = [ 0,  5, 10, 15, 16, 21, 26, 31, 32, 37, 42, 47, 48, 53, 58, 63,
 12,  1,  6, 11, 28, 17, 22, 27, 44, 33, 38, 43, 60, 49, 54, 59,
  8, 13,  2,  7, 24, 29, 18, 23, 40, 45, 34, 39, 56, 61, 50, 55,
  4,  9, 14,  3, 20, 25, 30, 19, 36, 41, 46, 35, 52, 57, 62, 51];
GIFT_RC = [0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F,
    0x1E, 0x3C, 0x39, 0x33, 0x27, 0x0E, 0x1D, 0x3A, 0x35, 0x2B,
    0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B, 0x17, 0x2E,
    0x1C, 0x38, 0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A,
    0x34, 0x29, 0x12, 0x24, 0x08, 0x11, 0x22, 0x04, 0x09, 0x13,
    0x26, 0x0c, 0x19, 0x32, 0x25, 0x0a, 0x15, 0x2a, 0x14, 0x28,
    0x10, 0x20];

#def bit_arr_to_hex_string(bit_arr):

def expand_key(k, r):
    expanded_key=[hex(k[0])[2:].zfill(16)+hex(k[1])[2:].zfill(16)];
    key = [int(expanded_key[0][i],16) for i in range(31,-1,-1)]
    #print(key);
    for i in range(0,r-1):
        temp_key=[0]*32;
        for i in range(0,32):
            temp_key[i] = key[(i+8)%32];
        for i in range(0,24):
            key[i] =  temp_key[i];
        key[24] = temp_key[27];
        key[25] = temp_key[24];
        key[26] = temp_key[25];
        key[27] = temp_key[26];
        key[28] = ((temp_key[28]&0xc)>>2) ^ ((temp_key[29]&0x3)<<2);
        key[29] = ((temp_key[29]&0xc)>>2) ^ ((temp_key[30]&0x3)<<2);
        key[30] = ((temp_key[30]&0xc)>>2) ^ ((temp_key[31]&0x3)<<2);
        key[31] = ((temp_key[31]&0xc)>>2) ^ ((temp_key[28]&0x3)<<2);
        #print(key);
        expanded_key.append("".join([hex(k)[2:] for k in key][::-1]));
    return expanded_key;
    
def enc_one_round(p,k,GIFT_RC_r):
    p_GIFT_S = [GIFT_S[int(p[i],16)] for i in range(0,16)]
    print(p_GIFT_S);
    p_GIFT_S_bits_array = [ [int(x) for x in np.binary_repr(y,4)] for y in p_GIFT_S]
    p_GIFT_S_bits = []
    for bit_arr in p_GIFT_S_bits_array: 
        p_GIFT_S_bits = p_GIFT_S_bits + bit_arr[::-1]
    #print(p_GIFT_S_bits);
    p_GIFT_P_bits = [0]*64;
    for i in range(0,64):
        p_GIFT_P_bits[GIFT_P[i]] = p_GIFT_S_bits[i];
    #print(p_GIFT_P_bits);
    key = [int(k[i],16) for i in range(0,32)]
    #print("Key");
    #print(key);
    key_bits_array = [ [int(x) for x in np.binary_repr(y,4)] for y in key] 
    key_bits = []
    for bit_arr in key_bits_array:
        key_bits = key_bits + bit_arr[::-1]
    #print(key_bits);
    key_bits_counter = 0;
    for i in range(0,16):
        p_GIFT_P_bits[4*i]  ^=  key_bits[key_bits_counter];
        p_GIFT_P_bits[4*i + 1]  ^= key_bits[key_bits_counter + 16];
        key_bits_counter += 1;
    p_GIFT_P_bits[3] ^= GIFT_RC_r & 0x1;
    p_GIFT_P_bits[7] ^= (GIFT_RC_r>>1) & 0x1;
    p_GIFT_P_bits[11] ^= (GIFT_RC_r>>2) & 0x1;
    p_GIFT_P_bits[15] ^= (GIFT_RC_r>>3) & 0x1;
    p_GIFT_P_bits[19] ^= (GIFT_RC_r>>4) & 0x1;
    p_GIFT_P_bits[23] ^= (GIFT_RC_r>>5) & 0x1;
    p_GIFT_P_bits[63] ^= 1;
    
    p_GIFT_P_bits_string = "".join([str(bit) for bit in p_GIFT_P_bits]);
    #print(p_GIFT_P_bits_string);
    plain_updated = ("".join([hex(int((p_GIFT_P_bits_string[i:i+4])[::-1],2))[2:] for i in range(0,len(p_GIFT_P_bits_string),4)]))[::-1] 
    return plain_updated;
   
def encrypt(p, ks):
    #x, y = p[0], p[1];
    x = hex(p)[2:].zfill(16); 
    for i in range(0,len(ks)):
        #print("round %d"%(i+1));
        #print(x);
        #print(ks[i]);
        x = enc_one_round(x[::-1], ks[i][::-1], GIFT_RC[i]);
        #print("Cipher Text is: " + x);
        #print("Updated Key is: " + ks[i+1])
    return int(x,16);
#def decrypt(c, ks):
    
#def check_testvector():
    
# # #plain = np.frombuffer(urandom(8*1),dtype=np.uint64);
# plain = int("c450c7727a9b8a7d",16);
# key = (int("bd91731eb6bc2713",16),int("a1f9f6ffc75044e7",16));
# #print(int(plain,16))
# #enc_one_round(int(plain,16),int(key,16))
# #enc_one_round(plain[::-1],key[::-1],0)
# key_expanded = expand_key(key,28);
# #rint(key_expanded)
# c = encrypt(plain,key_expanded);
# print(hex(c))
# #print(hex(plain[0]))
# # import os
# # keys = np.frombuffer(os.urandom(16*4),dtype=np.uint64).reshape(2,-1);
# # print(keys);
# # #print(list(zip(keys[0],keys[1])));
# # test_expansion = list(map(expand_key,list(zip(keys[0],keys[1])),[28]*len(keys[0])))
# # test_expansion_1 = list(map(expand_key,list(zip(keys[0],keys[1])),[25]*len(keys[0])))
# # #test_expansion = expand_key(keys,28);
# # print(test_expansion);
# # plain0 = np.frombuffer(os.urandom(8*4),dtype=np.uint64);
# # print(plain0);
# # ctdata0 = list(map(encrypt,plain0,test_expansion));
# # print(ctdata0);
# # ctdata0_1 = list(map(encrypt,plain0,test_expansion_1));
# # print(ctdata0_1);
    
def convert_to_binary_64_block(arr,WORD_SIZE=64):
  X = np.zeros((2 * WORD_SIZE,len(arr[0])),dtype=np.uint8);
  for i in range(2 * WORD_SIZE):
    index = i // WORD_SIZE;
    offset = WORD_SIZE - (i % WORD_SIZE) - 1;
    X[i] = (arr[index] >> offset) & 1;
  X = X.transpose();
  return(X);
    
def make_train_data_gift(n, nr, diff=0x0000000000000000):
  Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
  keys = np.frombuffer(urandom(16*n),dtype=np.uint64).reshape(2,-1); # 16*8*n bits with bock of 64
  plain0 = np.frombuffer(urandom(8*n),dtype=np.uint64);
  #plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
  #plain1l = plain0l ^ diff[0]; #plain1r = plain0r ^ diff[1];
  plain1 = plain0 ^ diff;
  num_rand_samples = np.sum(Y==0);
  plain1[Y==0] = np.frombuffer(urandom(8*num_rand_samples),dtype=np.uint64);
  #plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  
  ks = list(map(expand_key,list(zip(keys[0],keys[1])),[nr]*len(keys[0])));
  #ks = gi.expand_key(keys, nr);
  ctdata0 = np.array(list(map(encrypt,plain0,ks)),dtype=np.uint64);
  #ctdata0.astype(np.uint64);
  ctdata1 = np.array(list(map(encrypt,plain1,ks)),dtype=np.uint64);
  #ctdata1.astype(np.uint64);
  #ctdata0 = gi.encrypt(plain0, ks);
  #ctdata1= gi.encrypt(plain1, ks); np.binary_repr
  # print("plain0");
  # print(plain0)
  # print("plain1");
  # print(plain1)
  print("key");
  print(ks);
  # print("ctdata0");
  # print(ctdata0);
  # print("ctdata1");
  # print(ctdata1);
  X = convert_to_binary_64_block([ctdata0, ctdata1],64);
  #print(X,Y);
  return(X,Y);
#make_train_data_gift(2,28)
#sys.exit()
