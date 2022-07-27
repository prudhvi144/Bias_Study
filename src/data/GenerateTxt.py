#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Prudhvi Thirumalaraju
# Created Date: 07/22/2022
# version ='1.0'
# ---------------------------------------------------------------------------
"""Generate txt files for training and testing with the given distribution """
# ---------------------------------------------------------------------------



import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_implantation():

    df = pd.read_csv('./Bias EmbryoScope Data 4-29-22.csv', encoding='utf-8')
    ncols = len(df.columns)
    txt_implantation = df[["EmbryoScope Image ID", "PREG"]]
    print (txt_implantation)
    file_name = df["EmbryoScope Image ID"].tolist()
    impl = df["EmbryoScope Image ID"].tolist()
    file_name = txt_implantation.values.tolist()
    # print(file_name)
    root = './data/'
    df1 = txt_implantation[df["PREG"] == 0]
    file_name = df1.values.tolist()
    for f in file_name:
        print (f)
        with open('./txt/' + "_train" + ".txt", 'a') as the_file:
             the_file.write(root + f[0]+ " " + str(f[1]) + " " + '\n')

    # f = open('./txt/' + "_train" + ".txt", 'a')
    # f.writelines(['\n', str(df["EmbryoScope Image ID"]), ' ',str(df["Known Outcome? 1=Yes, 0=No"]) ])
    # f.close()

        # with open('./data/embryo/' + "_train" + ".txt", 'a') as the_file:
            # with open('../data/sd1/val.txt', 'a') as the_file:
            # the_file.write(data_dir_path+img_name+" "+img_name+'\n')
            # the_file.write(source_data_dir_path + classes + "/" + img_name + " " + str(int(classes) - 1) + " " + '0' + '\n')

if __name__ == '__main__':
    get_implantation()

