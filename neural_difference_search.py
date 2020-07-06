#test if rapid training runs using neural networks can be used to find good initial differences in Speck32/64
#idea: try all differences up to a certain weight and keep track of the performance level reached
import sys
import train_nets as tn
import speck as sp
import simon as si
#import gift as gi
import gift_64_opt as gi_64_opt
import gift_64_cofb as gi_64_cofb
import gimli_384
import numpy as np
#from os import urandom

from keras.models import Model
from sklearn.linear_model import Ridge

from random import randint
from collections import defaultdict
from math import log2

linear_model = Ridge(alpha=0.01);

#def WORD_SIZE():
#    return(64);

#first, we train a network to distinguish 3-round Speck with a randomly chosen input difference
#then, we use the penultimate layer output of that network to preprocess output data for other differences
#The preprocessed output for 1000 examples each is fed as training data to a single-layer perceptron
#That single-layer perceptron is then evaluated using 1000 validation samples
#The validation accuracy of the perceptron is taken as an indication of how much the differential distribution deviates from uniformity for that input difference


def train_preprocessor(n, nr, epochs, diff_in):
  
  #print(hex(diff_in[0]), hex(diff_in[1]));
  if (sys.argv[1]=="speck"):
      net = tn.make_resnet(depth=1,d1=64,d2=1024);# d1 must be 64 previously it was 1024
      net.compile(optimizer='adam',loss='mse',metrics=['acc']); 
      net.summary();
      X,Y = sp.make_train_data(n, nr, diff=diff_in);
  elif (sys.argv[1]=="simon"):
      net = tn.make_resnet(depth=1,d1=64,d2=1024); # d1 must be 64
      net.compile(optimizer='adam',loss='mse',metrics=['acc']);
      net.summary();
      X,Y = si.make_train_data(n, nr, diff=diff_in);
  elif (sys.argv[1]=="GIFT_64"):
      #net = tn.make_resnet(depth=2,d1=128,d2=1024,num_blocks=4,word_size=16); # d1 must be 128
      net = tn.make_resnet(depth=2,d1=64,d2=1024,num_blocks=4,word_size=int(16,2)); #word size decresed because of c0^c1
      net.compile(optimizer='adam',loss='mse',metrics=['acc']);
      net.summary();
      #X,Y = make_train_data_gift(n, nr, diff=diff_in);
      d = ((diff_in >> 48) & 0xffff, (diff_in >> 32) & 0xffff, (diff_in >> 16) & 0xffff, diff_in & 0xffff);
      #d = ((diff_in >> 32) & 0xffffffff, diff_in & 0xffffffff);
      X,Y = gi_64_opt.make_train_data(n, nr, diff=d);
  elif (sys.argv[1]=="GIFT_64_COFB"):
      net = tn.make_resnet(depth=2,d1=128,d2=1024,num_blocks=4,word_size=16); # d1 must be 128
      net.compile(optimizer='adam',loss='mse',metrics=['acc']);
      net.summary();
      #X,Y = make_train_data_gift(n, nr, diff=diff_in);
      d = ((diff_in >> 48) & 0xffff, (diff_in >> 32) & 0xffff, (diff_in >> 16) & 0xffff, diff_in & 0xffff);
      #d = ((diff_in >> 32) & 0xffffffff, diff_in & 0xffffffff);
      X,Y = gi_64_cofb.make_train_data(n, nr, diff=d);
  elif (sys.argv[1]=="GIMLI_384"):
      net = tn.make_resnet(depth=2,d1=384,d2=1024,num_blocks=12,word_size=int(32/2)); # word size decresed because of c0^c1
      net.compile(optimizer='adam',loss='mse',metrics=['acc']);
      net.summary();
      #X,Y = make_train_data_gift(n, nr, diff=diff_in);
      d = ((diff_in >> 352) & 0xffffffff,(diff_in >> 320) & 0xffffffff,(diff_in >> 288) & 0xffffffff,(diff_in >> 256) & 0xffffffff,(diff_in >> 224) & 0xffffffff,(diff_in >> 192) & 0xffffffff,(diff_in >> 160) & 0xffffffff,(diff_in >> 128) & 0xffffffff, (diff_in >> 96) & 0xffffffff, (diff_in >> 64) & 0xffffffff, (diff_in >> 32) & 0xffffffff, diff_in & 0xffffffff);
      #d = ((diff_in >> 32) & 0xffffffff, diff_in & 0xffffffff);
      X,Y = gimli_384.make_train_data(n, nr, diff=d);
      
  net.fit(X,Y,epochs=epochs, batch_size=5000,validation_split=0.1);
  net_pp = Model(inputs=net.layers[0].input, outputs=net.layers[-2].output);
  return(net_pp);

def evaluate_diff(diff, net_pp, nr=3, n=1000):
  if (diff == 0): return(0.0);
  d = (diff >> 16, diff & 0xffff);
  if (sys.argv[1]=="speck"):
    X,Y = sp.make_train_data(2*n, nr,diff=d);
  elif (sys.argv[1]=="simon"):
    X,Y = si.make_train_data(2*n, nr,diff=d);
  elif (sys.argv[1]=="GIFT_64"):
      #X,Y = make_train_data_gift(2*n, nr, diff=diff); 
      d = ((diff >> 48) & 0xffff, (diff >> 32) & 0xffff, (diff >> 16) & 0xffff, diff & 0xffff);
      #d = ((diff >> 32) & 0xffffffff, diff & 0xffffffff);
      X,Y = gi_64_opt.make_train_data(2*n, nr, diff=d);
  elif (sys.argv[1]=="GIFT_64_COFB"):
      d = ((diff >> 48) & 0xffff, (diff >> 32) & 0xffff, (diff >> 16) & 0xffff, diff & 0xffff);
      X,Y = gi_64_cofb.make_train_data(2*n, nr, diff=d);
  elif (sys.argv[1]=="GIMLI_384"):
      d = ((diff >> 352) & 0xffffffff,(diff >> 320) & 0xffffffff,(diff >> 288) & 0xffffffff,(diff >> 256) & 0xffffffff,(diff >> 224) & 0xffffffff,(diff >> 192) & 0xffffffff,(diff >> 160) & 0xffffffff,(diff >> 128) & 0xffffffff, (diff >> 96) & 0xffffffff, (diff >> 64) & 0xffffffff, (diff >> 32) & 0xffffffff, diff & 0xffffffff);
      X,Y = gimli_384.make_train_data(2*n, nr, diff=d);
  #perceptron.fit(Z[0:n],Y[0:n]);
  Z = net_pp.predict(X,batch_size=5000);
  linear_model.fit(Z[0:n],Y[0:n]);
  #val_acc = perceptron.score(Z[n:],Y[n:]);
  Y2 = linear_model.predict(Z[n:]);
  Y2bin = (Y2 > 0.5);
  val_acc = float(np.sum(Y2bin == Y[n:])) / n;
  return(val_acc);

#for a given difference, derive a guess how many rounds may be attackable
def extend_attack(diff, net_pp, nr, val_acc):
  print("Estimates of attack accuracy:");
  while(val_acc > 0.52):
    print(str(nr) + " rounds:" + str(val_acc));
    nr = nr + 1;
    val_acc = evaluate_diff(diff, net_pp, nr=nr, n=1000);

def greedy_optimizer_with_exploration(guess, f, n=2000, alpha=0.01, num_bits=32):
  best_guess = guess;
  best_val = f(guess); val = best_val;
  d = defaultdict(int)
  for i in range(n):
    d[guess] = d[guess] + 1; 
    r = randint(0, num_bits-1);
    guess_neu = guess ^ (1 << r);
    val_neu = f(guess_neu);
    if (val_neu > best_val):
      best_val = val_neu; best_guess = guess_neu;
      print(hex(best_guess), best_val);
    if (val_neu - alpha*log2(d[guess_neu]+1) > val - alpha*log2(d[guess]+1)):
      val = val_neu; guess = guess_neu;
  return(best_guess, best_val);

bits = 32;
if (sys.argv[1]=="GIFT_64" or sys.argv[1]=="GIFT_64_COFB"):
    bits = 64; 
elif (sys.argv[1]=="GIMLI_384" ):
    bits = 384;
#net_pp = train_preprocessor(2**18,3,5);
if (sys.argv[7]=="diff_random"):
    if(bits==32):
        difference = (randint(0,(2**16) - 1), randint(0,(2**16)-1));
    elif(bits==64):
        difference = randint(0,(2**64)-1)
        print("Difference chose is: " + hex(difference));
    elif(bits==384):
        difference=randint(0,(2**384)-1);
elif (sys.argv[7]=="diff_fix"):
    #diff_in = (int(0x0008),int(0x0000));
    if(bits==32):
        difference = (int(sys.argv[8],16),int(sys.argv[9],16));
    elif(bits==64):
        difference = int(sys.argv[8],16);

net_pp = train_preprocessor(int(sys.argv[2])**int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),difference);
#test_diff = int('0x00400000',16);
#test_diff = int('0x00010000',16);
#test_val_acc = evaluate_diff(test_diff, net_pp, 3)
#print("test_diff - diff: " + str(hex(test_diff)) + " Accuracy: " + str(test_val_acc)+"\n");
# diff, val_acc = greedy_optimizer_with_exploration(int(0x00010000), lambda x: evaluate_diff(x, net_pp, 3));
# print("After optimizer- diff: " + str(hex(diff)) + " Accuracy: " + str(val_acc)+"\n");
# extend_attack(diff, net_pp, 3, val_acc);
#sys.exit()
for i in range(int(sys.argv[6])):
  print("Run ",i,": ");
  #rand_int = randint(0,2**32-1); 
  diff, val_acc = greedy_optimizer_with_exploration(randint(0,(2**bits)-1), lambda x: evaluate_diff(x, net_pp, int(sys.argv[4])),num_bits=bits);
  print("After optimizer- diff: " + str(hex(diff)) + " Accuracy: " + str(val_acc)+"\n");
  extend_attack(diff, net_pp, int(sys.argv[4]), val_acc);
