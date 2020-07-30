from keras.models import load_model
import speck as sp
import simon as si
import gift_64_opt as gi_64_opt
import gift_64_cofb as gi_64_cofb
import gimli_384
import sys
import os
# load model
model = load_model(sys.argv[2]);
# summarize model.
file_name = os.path.basename(sys.argv[2]);
model.summary();
acc_count = 0;
for i in range(int(sys.argv[6])):
    if (sys.argv[1]=="speck"):
        #X_eval, Y_eval = sp.make_train_data(int(sys.argv[3])**int(sys.argv[4]), int((file_name.split('_',2))[1]));
        if (sys.argv[7]=="random"):
            X_eval, Y_eval = sp.make_train_data(int(sys.argv[3])**int(sys.argv[4]), int(sys.argv[5]),diff=(int(file_name.split('_')[7][1:-1].split(',')[0]),int(file_name.split('_')[7][1:-1].split(',')[1])),r_start=int(sys.argv[-1]));
        elif (sys.argv[7]=="no_random_default"):
            X_eval, Y_eval = sp.make_train_data_no_random(int(sys.argv[3])**int(sys.argv[4]), int(sys.argv[5]),diff=(int(file_name.split('_')[7][1:-1].split(',')[0]),int(file_name.split('_')[7][1:-1].split(',')[1])),r_start=int(sys.argv[-1]),output_Y=1);
        elif (sys.argv[7]=="no_random_diff"):
            X_eval, Y_eval = sp.make_train_data_no_random(int(sys.argv[3])**int(sys.argv[4]), int(sys.argv[5]),diff=(int(sys.argv[8],16),int(sys.argv[9],16)),r_start=int(sys.argv[-1]),output_Y=0);
            print('Using Diff: (' + str(sys.argv[8]) + ' , ' + str(sys.argv[9]) +')');

    elif (sys.argv[1]=="simon"):
        #X_eval, Y_eval = si.make_train_data(int(sys.argv[3])**int(sys.argv[4]), int((file_name.split('_',2))[1]));
        if (sys.argv[7]=="random"):
            X_eval, Y_eval = si.make_train_data(int(sys.argv[3])**int(sys.argv[4]), int(sys.argv[5]),diff=(int(file_name.split('_')[7][1:-1].split(',')[0]),int(file_name.split('_')[7][1:-1].split(',')[1])));
        elif (sys.argv[7]=="no_random_default"):
            X_eval, Y_eval = si.make_train_data_no_random(int(sys.argv[3])**int(sys.argv[4]), int(sys.argv[5]),diff=(int(file_name.split('_')[5][1:-1].split(',')[0]),int(file_name.split('_')[5][1:-1].split(',')[1])),output_Y=1);
            #X_eval, Y_eval = si.make_train_data_no_random(int(sys.argv[3])**int(sys.argv[4]), int(sys.argv[5]),diff=(int(245),int(433)));
        elif (sys.argv[7]=="no_random_diff"):
            X_eval, Y_eval = si.make_train_data_no_random(int(sys.argv[3])**int(sys.argv[4]), int(sys.argv[5]),diff=(int(sys.argv[8],16),int(sys.argv[9],16)),output_Y=0);
            print('Using Diff: (' + str(sys.argv[8]) + ' , ' + str(sys.argv[9]) +')');
    elif (sys.argv[1]=="GIFT_64_ENCRYPT" or sys.argv[1]=="GIFT_64_DECRYPT"):
        #X_eval, Y_eval = si.make_train_data(int(sys.argv[3])**int(sys.argv[4]), int((file_name.split('_',2))[1]));
        encrypt_data=1;
        if(sys.argv[1]=="GIFT_64_DECRYPT"):
          encrypt_data = 0;
        if (sys.argv[7]=="random"):
            X_eval, Y_eval = gi_64_opt.make_train_data(int(sys.argv[3])**int(sys.argv[4]), int(sys.argv[5]),diff=(int(file_name.split('_')[7][1:-1].split(',')[0]),int(file_name.split('_')[7][1:-1].split(',')[1]),int(file_name.split('_')[7][1:-1].split(',')[2]),int(file_name.split('_')[7][1:-1].split(',')[3])),r_start=int(sys.argv[-1]),encrypt_data=encrypt_data);
        elif (sys.argv[7]=="no_random_default"):
            X_eval, Y_eval = gi_64_opt.make_train_data_no_random(int(sys.argv[3])**int(sys.argv[4]), int(sys.argv[5]),diff=(int(file_name.split('_')[7][1:-1].split(',')[0]),int(file_name.split('_')[7][1:-1].split(',')[1]),int(file_name.split('_')[7][1:-1].split(',')[2]),int(file_name.split('_')[7][1:-1].split(',')[3])),output_Y=1,r_start=int(sys.argv[-1]),encrypt_data=encrypt_data);
            #X_eval, Y_eval = si.make_train_data_no_random(int(sys.argv[3])**int(sys.argv[4]), int(sys.argv[5]),diff=(int(245),int(433)));
        elif (sys.argv[7]=="no_random_diff"):
            X_eval, Y_eval = gi_64_opt.make_train_data_no_random(int(sys.argv[3])**int(sys.argv[4]), int(sys.argv[5]),diff=(int(sys.argv[8],16),int(sys.argv[9],16),int(sys.argv[10],16),int(sys.argv[11],16)),output_Y=0,r_start=int(sys.argv[-1]),encrypt_data=encrypt_data);
            print('Using Diff: (' + str(sys.argv[8]) + ' , ' + str(sys.argv[9]) + ' , ' + str(sys.argv[10]) + ' , ' + str(sys.argv[11])  +')');
    elif (sys.argv[1]=="GIFT_64_COFB"):
        #X_eval, Y_eval = si.make_train_data(int(sys.argv[3])**int(sys.argv[4]), int((file_name.split('_',2))[1]));
        if (sys.argv[7]=="random"):
            X_eval, Y_eval = gi_64_cofb.make_train_data(int(sys.argv[3])**int(sys.argv[4]), int(sys.argv[5]),diff=(int(file_name.split('_')[5][1:-1].split(',')[0]),int(file_name.split('_')[5][1:-1].split(',')[1]),int(file_name.split('_')[5][1:-1].split(',')[2]),int(file_name.split('_')[5][1:-1].split(',')[3])));
        elif (sys.argv[7]=="no_random_default"):
            X_eval, Y_eval = gi_64_cofb.make_train_data_no_random(int(sys.argv[3])**int(sys.argv[4]), int(sys.argv[5]),diff=(int(file_name.split('_')[5][1:-1].split(',')[0]),int(file_name.split('_')[5][1:-1].split(',')[1]),int(file_name.split('_')[5][1:-1].split(',')[2]),int(file_name.split('_')[5][1:-1].split(',')[3])),output_Y=1);
            #X_eval, Y_eval = si.make_train_data_no_random(int(sys.argv[3])**int(sys.argv[4]), int(sys.argv[5]),diff=(int(245),int(433)));
        elif (sys.argv[7]=="no_random_diff"):
            X_eval, Y_eval = gi_64_cofb.make_train_data_no_random(int(sys.argv[3])**int(sys.argv[4]), int(sys.argv[5]),diff=(int(sys.argv[8],16),int(sys.argv[9],16),int(sys.argv[10],16),int(sys.argv[11],16)),output_Y=0);
            print('Using Diff: (' + str(sys.argv[8]) + ' , ' + str(sys.argv[9]) + ' , ' + str(sys.argv[10]) + ' , ' + str(sys.argv[11])  +')');
    elif (sys.argv[1]=="GIMLI_384"):
        #X_eval, Y_eval = si.make_train_data(int(sys.argv[3])**int(sys.argv[4]), int((file_name.split('_',2))[1]));
        if (sys.argv[7]=="random"):
            X_eval, Y_eval = gimli_384.make_train_data(int(sys.argv[3])**int(sys.argv[4]), int(sys.argv[5]),diff=(int(file_name.split('_')[5][1:-1].split(',')[0]),int(file_name.split('_')[5][1:-1].split(',')[1]),int(file_name.split('_')[5][1:-1].split(',')[2]),int(file_name.split('_')[5][1:-1].split(',')[3]),int(file_name.split('_')[5][1:-1].split(',')[4]),int(file_name.split('_')[5][1:-1].split(',')[5]),int(file_name.split('_')[5][1:-1].split(',')[6]),int(file_name.split('_')[5][1:-1].split(',')[7]),int(file_name.split('_')[5][1:-1].split(',')[8]),int(file_name.split('_')[5][1:-1].split(',')[9]),int(file_name.split('_')[5][1:-1].split(',')[10]),int(file_name.split('_')[5][1:-1].split(',')[11])));
        elif (sys.argv[7]=="no_random_default"):
            X_eval, Y_eval = gimli_384.make_train_data_no_random(int(sys.argv[3])**int(sys.argv[4]), int(sys.argv[5]),diff=(int(file_name.split('_')[5][1:-1].split(',')[0]),int(file_name.split('_')[5][1:-1].split(',')[1]),int(file_name.split('_')[5][1:-1].split(',')[2]),int(file_name.split('_')[5][1:-1].split(',')[3]),int(file_name.split('_')[5][1:-1].split(',')[4]),int(file_name.split('_')[5][1:-1].split(',')[5]),int(file_name.split('_')[5][1:-1].split(',')[6]),int(file_name.split('_')[5][1:-1].split(',')[7]),int(file_name.split('_')[5][1:-1].split(',')[8]),int(file_name.split('_')[5][1:-1].split(',')[9]),int(file_name.split('_')[5][1:-1].split(',')[10]),int(file_name.split('_')[5][1:-1].split(',')[11])),output_Y=1);
            #X_eval, Y_eval = si.make_train_data_no_random(int(sys.argv[3])**int(sys.argv[4]), int(sys.argv[5]),diff=(int(245),int(433)));
        elif (sys.argv[7]=="no_random_diff"):
            X_eval, Y_eval = gimli_384.make_train_data_no_random(int(sys.argv[3])**int(sys.argv[4]), int(sys.argv[5]),diff=(int(sys.argv[8],16),int(sys.argv[9],16),int(sys.argv[10],16),int(sys.argv[11],16),int(sys.argv[12],16),int(sys.argv[13],16),int(sys.argv[14],16),int(sys.argv[15],16),int(sys.argv[16],16),int(sys.argv[17],16),int(sys.argv[18],16),int(sys.argv[19],16)),output_Y=0);
            print('Using Diff: (' + str(sys.argv[8]) + ' , ' + str(sys.argv[9]) + ' , ' + str(sys.argv[10]) + ' , ' + str(sys.argv[11]) + ' , ' + str(sys.argv[12])+ ' , ' + str(sys.argv[13])+ ' , ' + str(sys.argv[14])+ ' , ' + str(sys.argv[15])+ ' , ' + str(sys.argv[16])+ ' , ' + str(sys.argv[17])+ ' , ' + str(sys.argv[18])+ ' , ' + str(sys.argv[19]) +')');
    # evaluate the model
    score = model.evaluate(X_eval, Y_eval)
    print("Run %d : %s: %.2f%%" % (i+1, model.metrics_names[1], score[1]*100))
    if (score[1]>.51):
        acc_count = acc_count + 1;
print("Total Count for acc > .51 : %d"%(acc_count));

#call simon ./simon/nets/..hd5 2 12 8 1 random/no_random