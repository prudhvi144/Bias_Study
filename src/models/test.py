#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Prudhvi Thirumalaraju   Line 3
# Created Date: 08/11/2022
# version ='1.0'
# ---------------------------------------------------------------------------
""" Testing scripts """
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import numpy as np
import csv
import yaml
import numpy as np
from testing import main
CONFIG_PATH = "../data/"

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config

def get_model(file,sampling):
    print (file)
    #read the files in a folder
    path = "../../reports/logs/"+ sampling+ "/train_"+file+"/"
    dir_list = os.listdir(path)
    log ="log.txt"
    a = []
    dicts = {}
    for files in dir_list:
        # print(files)
        string1 ="Final Model | Val loss:"
        with open(os.path.join(path, files ,log),"r") as f:
            for line in f:

                if string1 in line:
                    print(line)
                    x = line.split("|")[1].split(": ")[1]
                    a.append(float(x))
                    print(float(x))

        dicts[files] = x
    print (dicts)
    df = pd.DataFrame.from_dict(dicts, orient='index',columns=['value']).sort_values(by=['value'])
    g =df.iloc[0].name
    model_path = "../../reports/models/"+ sampling+ "/train_"+file+"/"+g+"/best_model.pth.tar"
    print(model_path)
    return(model_path)



def get_data(file,sampling):

    path = "../../data/txt_files/" + sampling + "/" + file + "/Test"
    dir_list = os.listdir(path)
    paths = []

    for files in dir_list:
        # print(files)
        paths.append("../../data/txt_files/" + sampling + "/" + file + "/Test/"+files+"/"+"Test.txt")
    return(paths)


    pass


# for data_path in paths:
#     test()
#     pass


if __name__ == '__main__':


    config = load_config("../data/my_config.yaml")
    print(config)

    # for samp in config['Sampling_char']:
    #    for i in config["Bias_char"]:
    #     p = get_model(i, samp)
    #     x = get_data(i,samp)
    #     for j in x:
    #         print(j)
    #         main(str(p),j)
    # samp = 'stratfied_bins'
    # samp = 'stratfied'
    samp = 'balanced_bins'
    # samp = 'residual'
    # samp = 'residual_bins'
    # samp = 'residual'
    logs_path = "../../reports/testing/"+ samp

    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    with open(logs_path + '/' +samp+ '_results.csv',
              mode='w') as file:
        csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        csv_writer.writerow(["Categories","Sub_Categories", "Accuracy", "Reference Accuracy" , "Loss" , "Precision","Recall","F1-score", "tn" , "fp" , "fn ", "tp" ,"AUC"])

    for i in config["Bias_numerical"]:

        try:
            p = get_model(i, samp)
        except:
            pass


        x = get_data(i, samp)
        for j in x:
            print(j)
            main(str(p), j)

