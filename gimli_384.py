#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 09:33:02 2020

@author: tarunyadav
"""

import numpy as np
from os import urandom
#import sys


def colperm(S,j):
    # S_2 = 0x00000000;
    # S_1 = 0x00000000;
    # S_0 = 0x00000000;
    # x = (S[0][j] << 24) | (S[0][j] >> 8);
    # y = (S[1][j] << 9) | (S[1][j] >> 23);
    # z = S[2][0];
    # print("xyz are");
    # print(hex(x[0]));
    # print(hex(y[0]));
    # print(hex(z[0]));
    # print("colperm");
    # print(hex(S[0][j][0]));
    # print(hex(S[1][j][0]));
    # print(hex(S[2][j][0]));
    # S_2 = ((S[0][j] << 24) | (S[0][j] >> 8)) ^ ((S[2][j])<<1) ^ ((((S[1][j] << 9) | (S[1][j] >> 23)) & (S[2][j]))<<2);
    # S_1 = ((S[1][j] << 9) | (S[1][j] >> 23)) ^ ((S[0][j] << 24) | (S[0][j] >> 8)) ^ ((((S[0][j] << 24) | (S[0][j] >> 8)) | (S[2][j])) << 1);
    # S_0 = (S[2][j]) ^ ((S[1][j] << 9) | (S[1][j] >> 23)) ^ ((((S[0][j] << 24) | (S[0][j] >> 8))& ((S[1][j] << 9) | (S[1][j] >> 23))) << 3);
    return((S[2][j]) ^ ((S[1][j] << 9) | (S[1][j] >> 23)) ^ ((((S[0][j] << 24) | (S[0][j] >> 8)) & ((S[1][j] << 9) | (S[1][j] >> 23))) << 3),((S[1][j] << 9) | (S[1][j] >> 23)) ^ ((S[0][j] << 24) | (S[0][j] >> 8)) ^ ((((S[0][j] << 24) | (S[0][j] >> 8)) | (S[2][j])) << 1),((S[0][j] << 24) | (S[0][j] >> 8)) ^ ((S[2][j])<<1) ^ ((((S[1][j] << 9) | (S[1][j] >> 23)) & (S[2][j]))<<2));

def enc_one_round(S_arr, r):
    S = np.copy(S_arr);
    # print("inital S");
    # print([hex(S[i][j][0])[2:].zfill(8) for i in range(0,3) for j in range(0,4) ]);
    ##To make it as per NIST GIFT
    S[0][0] = np.uint32(S[0][0] & 0xff)<<24 | np.uint32((S[0][0]>>8)&0xff)<<16 | np.uint32((S[0][0]>>16)&0xff)<<8 | np.uint32((S[0][0]>>24)&0xff);
    S[0][1] = np.uint32(S[0][1] & 0xff)<<24 | np.uint32((S[0][1]>>8)&0xff)<<16 | np.uint32((S[0][1]>>16)&0xff)<<8 | np.uint32((S[0][1]>>24)&0xff);
    S[0][2] = np.uint32(S[0][2] & 0xff)<<24 | np.uint32((S[0][2]>>8)&0xff)<<16 | np.uint32((S[0][2]>>16)&0xff)<<8 | np.uint32((S[0][2]>>24)&0xff);
    S[0][3] = np.uint32(S[0][3] & 0xff)<<24 | np.uint32((S[0][3]>>8)&0xff)<<16 | np.uint32((S[0][3]>>16)&0xff)<<8 | np.uint32((S[0][3]>>24)&0xff);
    S[1][0] = np.uint32(S[1][0] & 0xff)<<24 | np.uint32((S[1][0]>>8)&0xff)<<16 | np.uint32((S[1][0]>>16)&0xff)<<8 | np.uint32((S[1][0]>>24)&0xff);
    S[1][1] = np.uint32(S[1][1] & 0xff)<<24 | np.uint32((S[1][1]>>8)&0xff)<<16 | np.uint32((S[1][1]>>16)&0xff)<<8 | np.uint32((S[1][1]>>24)&0xff);
    S[1][2] = np.uint32(S[1][2] & 0xff)<<24 | np.uint32((S[1][2]>>8)&0xff)<<16 | np.uint32((S[1][2]>>16)&0xff)<<8 | np.uint32((S[1][2]>>24)&0xff);
    S[1][3] = np.uint32(S[1][3] & 0xff)<<24 | np.uint32((S[1][3]>>8)&0xff)<<16 | np.uint32((S[1][3]>>16)&0xff)<<8 | np.uint32((S[1][3]>>24)&0xff);
    S[2][0] = np.uint32(S[2][0] & 0xff)<<24 | np.uint32((S[2][0]>>8)&0xff)<<16 | np.uint32((S[2][0]>>16)&0xff)<<8 | np.uint32((S[2][0]>>24)&0xff);
    S[2][1] = np.uint32(S[2][1] & 0xff)<<24 | np.uint32((S[2][1]>>8)&0xff)<<16 | np.uint32((S[2][1]>>16)&0xff)<<8 | np.uint32((S[2][1]>>24)&0xff);
    S[2][2] = np.uint32(S[2][2] & 0xff)<<24 | np.uint32((S[2][2]>>8)&0xff)<<16 | np.uint32((S[2][2]>>16)&0xff)<<8 | np.uint32((S[2][2]>>24)&0xff);
    S[2][3] = np.uint32(S[2][3] & 0xff)<<24 | np.uint32((S[2][3]>>8)&0xff)<<16 | np.uint32((S[2][3]>>16)&0xff)<<8 | np.uint32((S[2][3]>>24)&0xff);
    S[:,0] = colperm(S,0);
    #print([hex(S[i][j][0])[2:].zfill(8) for i in range(0,3) for j in range(0,4) ]);
    S[:,1] = colperm(S,1);
    S[:,2] = colperm(S,2);
    S[:,3] = colperm(S,3);
    S[0][0] = np.uint32(S[0][0] & 0xff)<<24 | np.uint32((S[0][0]>>8)&0xff)<<16 | np.uint32((S[0][0]>>16)&0xff)<<8 | np.uint32((S[0][0]>>24)&0xff);
    S[0][1] = np.uint32(S[0][1] & 0xff)<<24 | np.uint32((S[0][1]>>8)&0xff)<<16 | np.uint32((S[0][1]>>16)&0xff)<<8 | np.uint32((S[0][1]>>24)&0xff);
    S[0][2] = np.uint32(S[0][2] & 0xff)<<24 | np.uint32((S[0][2]>>8)&0xff)<<16 | np.uint32((S[0][2]>>16)&0xff)<<8 | np.uint32((S[0][2]>>24)&0xff);
    S[0][3] = np.uint32(S[0][3] & 0xff)<<24 | np.uint32((S[0][3]>>8)&0xff)<<16 | np.uint32((S[0][3]>>16)&0xff)<<8 | np.uint32((S[0][3]>>24)&0xff);
    S[1][0] = np.uint32(S[1][0] & 0xff)<<24 | np.uint32((S[1][0]>>8)&0xff)<<16 | np.uint32((S[1][0]>>16)&0xff)<<8 | np.uint32((S[1][0]>>24)&0xff);
    S[1][1] = np.uint32(S[1][1] & 0xff)<<24 | np.uint32((S[1][1]>>8)&0xff)<<16 | np.uint32((S[1][1]>>16)&0xff)<<8 | np.uint32((S[1][1]>>24)&0xff);
    S[1][2] = np.uint32(S[1][2] & 0xff)<<24 | np.uint32((S[1][2]>>8)&0xff)<<16 | np.uint32((S[1][2]>>16)&0xff)<<8 | np.uint32((S[1][2]>>24)&0xff);
    S[1][3] = np.uint32(S[1][3] & 0xff)<<24 | np.uint32((S[1][3]>>8)&0xff)<<16 | np.uint32((S[1][3]>>16)&0xff)<<8 | np.uint32((S[1][3]>>24)&0xff);
    S[2][0] = np.uint32(S[2][0] & 0xff)<<24 | np.uint32((S[2][0]>>8)&0xff)<<16 | np.uint32((S[2][0]>>16)&0xff)<<8 | np.uint32((S[2][0]>>24)&0xff);
    S[2][1] = np.uint32(S[2][1] & 0xff)<<24 | np.uint32((S[2][1]>>8)&0xff)<<16 | np.uint32((S[2][1]>>16)&0xff)<<8 | np.uint32((S[2][1]>>24)&0xff);
    S[2][2] = np.uint32(S[2][2] & 0xff)<<24 | np.uint32((S[2][2]>>8)&0xff)<<16 | np.uint32((S[2][2]>>16)&0xff)<<8 | np.uint32((S[2][2]>>24)&0xff);
    S[2][3] = np.uint32(S[2][3] & 0xff)<<24 | np.uint32((S[2][3]>>8)&0xff)<<16 | np.uint32((S[2][3]>>16)&0xff)<<8 | np.uint32((S[2][3]>>24)&0xff);
    if (r % 4 == 0):
        Temp = np.copy(S[0][1]);
        S[0][1] = S[0][0];
        S[0][0] = Temp;
        Temp = np.copy(S[0][3]);
        S[0][3] = S[0][2];
        S[0][2] = Temp;
    elif (r % 4 == 2):
        Temp = np.copy(S[0][2]);
        S[0][2] = S[0][0];
        S[0][0] = Temp;
        Temp = np.copy(S[0][3]);
        S[0][3] = S[0][1];
        S[0][1] = Temp;
    if (r % 4 == 0):
        S[0][0] = np.uint32(S[0][0] & 0xff)<<24 | np.uint32((S[0][0]>>8)&0xff)<<16 | np.uint32((S[0][0]>>16)&0xff)<<8 | np.uint32((S[0][0]>>24)&0xff);
        S[0][0] ^= 0x9e377900 ^ r;
        S[0][0] = np.uint32(S[0][0] & 0xff)<<24 | np.uint32((S[0][0]>>8)&0xff)<<16 | np.uint32((S[0][0]>>16)&0xff)<<8 | np.uint32((S[0][0]>>24)&0xff);
    
    return(S);
    
def encrypt(p, r):
    S_final = np.copy([[p[0],p[1],p[2],p[3]],[p[4],p[5],p[6],p[7]],[p[8],p[9],p[10],p[11]]]);
    # print("S _Final");
    # print(S_final);
    # print("S _Final");
    for i in range(r,0,-1):
        S_final = enc_one_round(S_final, i);
    #C = S_final.reshape(1,-1);
    C = np.array([S_final[0][0],S_final[0][1],S_final[0][2],S_final[0][3],S_final[1][0],S_final[1][1],S_final[1][2],S_final[1][3],S_final[2][0],S_final[2][1],S_final[2][2],S_final[2][3]],dtype=np.uint32);
   
    return(C);

# key_test = np.array([[48529, 48529], [29470, 29470], [46780, 46780], [10003, 10003], [41465, 41465], [63231, 63231], [51024, 51024], [17639, 17639]],dtype=np.uint16)
# #plain_text = np.array([[63693, 17070], [ 1784, 58987], [ 4727, 51159], [28008, 28338]],dtype=np.uint16)
# # #plain_text = np.array([[3293628274, 3293628274], [ 2057013885, 2057013885]],dtype=np.uint32)
# plain_text = np.array([[50256, 50256], [ 51058, 51058],  [ 31387, 31387],  [ 35453, 35453]],dtype=np.uint16)

# # #print([hex(plain_text_i[0])[2:].zfill(8) for plain_text_i in plain_text ]);
# print([hex(plain_text_i[0])[2:].zfill(4) for plain_text_i in plain_text ]);

# expanded_key = expand_key(key_test,28);
# print([hex(expanded_key_i[j][0])[2:].zfill(4) for expanded_key_i in expanded_key for j in range(0,len(expanded_key_i))]);
# # #ctdata0_0, ctdata0_1, ctdata1_0, ctdata1_1 = encrypt((plain_text[0], plain_text[1], plain_text[0], plain_text[1]), expanded_key);
# ctdata0_0, ctdata0_1, ctdata0_2, ctdata0_3 = encrypt((plain_text[0], plain_text[1], plain_text[2], plain_text[3]), expanded_key);
# print("cdata0");
# # #print([hex(ctdata0_0[0])[2:].zfill(4), hex(ctdata0_1[0])[2:], hex(ctdata1_0[0])[2:], hex(ctdata1_1[0])[2:]]);
# print([hex(ctdata0_0[0])[2:].zfill(4), hex(ctdata0_1[0])[2:], hex(ctdata0_2[0])[2:], hex(ctdata0_3[0])[2:]]);

 
def convert_to_binary_384_block(arr,WORD_SIZE=32,NO_OF_WORDS=24):
  X = np.zeros((NO_OF_WORDS * WORD_SIZE,len(arr[0])),dtype=np.uint8);
  for i in range(NO_OF_WORDS * WORD_SIZE):
    index = i // WORD_SIZE;
    offset = WORD_SIZE - (i % WORD_SIZE) - 1;
    X[i] = (arr[index] >> offset) & 1;
  X = X.transpose();
  return(X);


  
def make_train_data(n, nr, diff=(0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000)):
    Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
    
    plain0_0 = np.frombuffer(urandom(4*n),dtype=np.uint32);
    plain0_1 = np.frombuffer(urandom(4*n),dtype=np.uint32);
    plain0_2 = np.frombuffer(urandom(4*n),dtype=np.uint32);
    plain0_3 = np.frombuffer(urandom(4*n),dtype=np.uint32);
    plain0_4 = np.frombuffer(urandom(4*n),dtype=np.uint32);
    plain0_5 = np.frombuffer(urandom(4*n),dtype=np.uint32);
    plain0_6 = np.frombuffer(urandom(4*n),dtype=np.uint32);
    plain0_7 = np.frombuffer(urandom(4*n),dtype=np.uint32);
    plain0_8 = np.frombuffer(urandom(4*n),dtype=np.uint32);
    plain0_9 = np.frombuffer(urandom(4*n),dtype=np.uint32);
    plain0_10 = np.frombuffer(urandom(4*n),dtype=np.uint32);
    plain0_11 = np.frombuffer(urandom(4*n),dtype=np.uint32);
    
    plain1_0 = plain0_0 ^ diff[0];
    plain1_1 = plain0_1 ^ diff[1];
    plain1_2 = plain0_2 ^ diff[2];
    plain1_3 = plain0_3 ^ diff[3];
    plain1_4 = plain0_4 ^ diff[4];
    plain1_5 = plain0_5 ^ diff[5];
    plain1_6 = plain0_6 ^ diff[6];
    plain1_7 = plain0_7 ^ diff[7];
    plain1_8 = plain0_8 ^ diff[8];
    plain1_9 = plain0_9 ^ diff[9];
    plain1_10 = plain0_10 ^ diff[10];
    plain1_11 = plain0_11 ^ diff[11];
    
    
    num_rand_samples = np.sum(Y==0);
    plain1_0[Y==0] = np.frombuffer(urandom(4*num_rand_samples),dtype=np.uint32);
    plain1_1[Y==0] = np.frombuffer(urandom(4*num_rand_samples),dtype=np.uint32);
    plain1_2[Y==0] = np.frombuffer(urandom(4*num_rand_samples),dtype=np.uint32);
    plain1_3[Y==0] = np.frombuffer(urandom(4*num_rand_samples),dtype=np.uint32);
    plain1_4[Y==0] = np.frombuffer(urandom(4*num_rand_samples),dtype=np.uint32);
    plain1_5[Y==0] = np.frombuffer(urandom(4*num_rand_samples),dtype=np.uint32);
    plain1_6[Y==0] = np.frombuffer(urandom(4*num_rand_samples),dtype=np.uint32);
    plain1_7[Y==0] = np.frombuffer(urandom(4*num_rand_samples),dtype=np.uint32);
    plain1_8[Y==0] = np.frombuffer(urandom(4*num_rand_samples),dtype=np.uint32);
    plain1_9[Y==0] = np.frombuffer(urandom(4*num_rand_samples),dtype=np.uint32);
    plain1_10[Y==0] = np.frombuffer(urandom(4*num_rand_samples),dtype=np.uint32);
    plain1_11[Y==0] = np.frombuffer(urandom(4*num_rand_samples),dtype=np.uint32);
  
  
  # print plain and key
    # print("plain0");
    # print("".join([hex(plain_text_i[0])[2:].zfill(8) for plain_text_i in [plain0_0, plain0_1, plain0_2, plain0_3,plain0_4, plain0_5, plain0_6, plain0_7,plain0_8, plain0_9, plain0_10, plain0_11] ]));
    # print([hex(plain_text_i[0])[2:].zfill(8) for plain_text_i in [plain0_0, plain0_1, plain0_2, plain0_3,plain0_4, plain0_5, plain0_6, plain0_7,plain0_8, plain0_9, plain0_10, plain0_11] ]);
    # print("plain1");
    # print([hex(plain_text_i[0])[2:].zfill(8) for plain_text_i in [plain1_0, plain1_1, plain1_2, plain1_3,plain1_4, plain1_5, plain1_6, plain1_7,plain1_8, plain1_9, plain1_10, plain1_11] ]);
    # print("".join([hex(plain_text_i[0])[2:].zfill(8) for plain_text_i in [plain1_0, plain1_1, plain1_2, plain1_3,plain1_4, plain1_5, plain1_6, plain1_7,plain1_8, plain1_9, plain1_10, plain1_11] ]));

    cdata0 = encrypt((plain0_0, plain0_1, plain0_2, plain0_3,plain0_4, plain0_5, plain0_6, plain0_7,plain0_8, plain0_9, plain0_10, plain0_11),nr);
    cdata1 = encrypt((plain1_0, plain1_1, plain1_2, plain1_3,plain1_4, plain1_5, plain1_6, plain1_7,plain1_8, plain1_9, plain1_10, plain1_11),nr);
  #X is 284 * 2 length
  #X = convert_to_binary_384_block(np.concatenate([cdata0,cdata1]),32,24);
    X = convert_to_binary_384_block(np.array(cdata0^cdata1),32,12);
  #print(X,Y);
  #return(X,Y);
  
  #print ciphertext
    # print("cdata0");
    # print([hex(cipher_text_i[0])[2:].zfill(8) for cipher_text_i in cdata0 ]);
    # print("cdata1");
    # print([hex(cipher_text_i[0])[2:].zfill(8) for cipher_text_i in cdata1 ]);
    # print(X)
    return (X,Y);
# make_train_data(1,24)
# sys.exit() 
def make_train_data_no_random(n, nr, diff=(0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000),output_Y=1):
  Y = np.frombuffer(urandom(n), dtype=np.uint8);
  if (output_Y==0):
    Y = (Y & 0);
  elif (output_Y==1):
    Y = (Y & 1) | 1;

  plain0_0 = np.frombuffer(urandom(4*n),dtype=np.uint32);
  plain0_1 = np.frombuffer(urandom(4*n),dtype=np.uint32);
  plain0_2 = np.frombuffer(urandom(4*n),dtype=np.uint32);
  plain0_3 = np.frombuffer(urandom(4*n),dtype=np.uint32);
  plain0_4 = np.frombuffer(urandom(4*n),dtype=np.uint32);
  plain0_5 = np.frombuffer(urandom(4*n),dtype=np.uint32);
  plain0_6 = np.frombuffer(urandom(4*n),dtype=np.uint32);
  plain0_7 = np.frombuffer(urandom(4*n),dtype=np.uint32);
  plain0_8 = np.frombuffer(urandom(4*n),dtype=np.uint32);
  plain0_9 = np.frombuffer(urandom(4*n),dtype=np.uint32);
  plain0_10 = np.frombuffer(urandom(4*n),dtype=np.uint32);
  plain0_11 = np.frombuffer(urandom(4*n),dtype=np.uint32);

  plain1_0 = plain0_0 ^ diff[0];
  plain1_1 = plain0_1 ^ diff[1];
  plain1_2 = plain0_2 ^ diff[2];
  plain1_3 = plain0_3 ^ diff[3];
  plain1_4 = plain0_4 ^ diff[4];
  plain1_5 = plain0_5 ^ diff[5];
  plain1_6 = plain0_6 ^ diff[6];
  plain1_7 = plain0_7 ^ diff[7];
  plain1_8 = plain0_8 ^ diff[8];
  plain1_9 = plain0_9 ^ diff[9];
  plain1_10 = plain0_10 ^ diff[10];
  plain1_11 = plain0_11 ^ diff[11];
  
  
  
  #print plain and key
  # print("plain0");
  # print([hex(plain_text_i[0])[2:].zfill(8) for plain_text_i in [plain0_0, plain0_1, plain0_2, plain0_3,plain0_4, plain0_5, plain0_6, plain0_7,plain0_8, plain0_9, plain0_10, plain0_11] ]);
  # print([hex(plain_text_i[0])[2:].zfill(8) for plain_text_i in [plain1_0, plain1_1, plain1_2, plain1_3,plain1_4, plain1_5, plain1_6, plain1_7,plain1_8, plain1_9, plain1_10, plain1_11] ]);

  cdata0 = encrypt((plain0_0, plain0_1, plain0_2, plain0_3,plain0_4, plain0_5, plain0_6, plain0_7,plain0_8, plain0_9, plain0_10, plain0_11),nr);
  cdata1 = encrypt((plain1_0, plain1_1, plain1_2, plain1_3,plain1_4, plain1_5, plain1_6, plain1_7,plain1_8, plain1_9, plain1_10, plain1_11),nr);
  
  #X = convert_to_binary_384_block(np.concatenate([cdata0,cdata1]),32,24);
  X = convert_to_binary_384_block(np.array(cdata0^cdata1),32,12);
  #print(X,Y);
  #return(X,Y);
  
  #print ciphertext
  # print("cdata0");
  # print([hex(cipher_text_i[0])[2:].zfill(8) for cipher_text_i in cdata0 ]);
  # print("cdata1");
  # print([hex(cipher_text_i[0])[2:].zfill(8) for cipher_text_i in cdata1]);
  return (X,Y);
# Without Comments
# def convert_to_binary_64_block(arr,WORD_SIZE=16):
#   X = np.zeros((8 * WORD_SIZE,len(arr[0])),dtype=np.uint8);
#   for i in range(8 * WORD_SIZE):
#     index = i // WORD_SIZE;
#     offset = WORD_SIZE - (i % WORD_SIZE) - 1;
#     X[i] = (arr[index] >> offset) & 1;
#   X = X.transpose();
#   return(X);

# def make_train_data_gift_opt(n, nr, diff=(0x0000,0x0000,0x0000,0x0000)):
#   Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
#   keys = np.frombuffer(urandom(16*n),dtype=np.uint16).reshape(8,-1); # 16*8*n no of key bits
#   plain0_0 = np.frombuffer(urandom(2*n),dtype=np.uint16);
#   plain0_1 = np.frombuffer(urandom(2*n),dtype=np.uint16);
#   plain0_2 = np.frombuffer(urandom(2*n),dtype=np.uint16);
#   plain0_3 = np.frombuffer(urandom(2*n),dtype=np.uint16);
#   plain1_0 = plain0_0 ^ diff[0];
#   plain1_1 = plain0_1 ^ diff[1];
#   plain1_2 = plain0_2 ^ diff[2];
#   plain1_3 = plain0_3 ^ diff[3];
#   num_rand_samples = np.sum(Y==0);
#   plain1_0[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
#   plain1_1[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
#   plain1_2[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
#   plain1_3[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
#   ks = expand_key(keys, nr);
#   ctdata0_0, ctdata0_1, ctdata0_2, ctdata0_3 = encrypt((plain0_0, plain0_1, plain0_2, plain0_3), ks);
#   ctdata1_0, ctdata1_1, ctdata1_2, ctdata1_3 = encrypt((plain1_0, plain1_1, plain1_2, plain1_3), ks);
#   X = convert_to_binary_64_block([ctdata0_0, ctdata0_1, ctdata0_2, ctdata0_3, ctdata1_0, ctdata1_1, ctdata1_2, ctdata1_3],16);
#   return (X,Y);











