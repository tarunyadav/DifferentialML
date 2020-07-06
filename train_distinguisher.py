import train_nets as tn
import sys
# simon 100 11 2 1024 2 26 2 24 0x0000 0x0020
# GIFT_64 100 5 2 1024 2 18 2 16 0x0000 0x0000 0x2000 0x0007
if ("GIFT_64" in sys.argv[1]):
    tn.train_speck_distinguisher(cipher=sys.argv[1],num_epochs=int(sys.argv[2]),num_rounds=int(sys.argv[3]),depth=int(sys.argv[4]),neurons=int(sys.argv[5]),data_train=int(sys.argv[6])**int(sys.argv[7]),data_test=int(sys.argv[8])**int(sys.argv[9]),difference=(int(sys.argv[10],16),int(sys.argv[11],16),int(sys.argv[12],16),int(sys.argv[13],16)),start_round=int(sys.argv[14]), pre_trained_model=sys.argv[15]);
elif (sys.argv[1]=="GIMLI_384"):
    tn.train_speck_distinguisher(cipher=sys.argv[1],num_epochs=int(sys.argv[2]),num_rounds=int(sys.argv[3]),depth=int(sys.argv[4]),neurons=int(sys.argv[5]),data_train=int(sys.argv[6])**int(sys.argv[7]),data_test=int(sys.argv[8])**int(sys.argv[9]),difference=(int(sys.argv[10],16),int(sys.argv[11],16),int(sys.argv[12],16),int(sys.argv[13],16),int(sys.argv[14],16),int(sys.argv[15],16),int(sys.argv[16],16),int(sys.argv[17],16),int(sys.argv[18],16),int(sys.argv[19],16),int(sys.argv[20],16),int(sys.argv[21],16)));
else:
    tn.train_speck_distinguisher(cipher=sys.argv[1],num_epochs=int(sys.argv[2]),num_rounds=int(sys.argv[3]),depth=int(sys.argv[4]),neurons=int(sys.argv[5]),data_train=int(sys.argv[6])**int(sys.argv[7]),data_test=int(sys.argv[8])**int(sys.argv[9]),difference=(int(sys.argv[10],16),int(sys.argv[11],16)),start_round=int(sys.argv[12]));