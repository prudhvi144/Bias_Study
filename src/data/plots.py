#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Prudhvi Thirumalaraju   Line 3
# Created Date: 07/22/2022
# version ='1.0'
# ---------------------------------------------------------------------------
""" Training scripts """
# ---------------------------------------------------------------------------
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import numpy as np

import yaml


CONFIG_PATH = "./"


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config

def reed_pand(bias_char):

    df_train = pd.read_csv('../../data/txt_files/'+bias_char+"/" + 'Train.csv',delimiter=',', encoding='utf-8',header=None)
    df_train.columns = ["A", "B", "C"]
    df_val = pd.read_csv('../../data/txt_files/' + bias_char + "/" + 'Val.csv', delimiter=',', encoding='utf-8',header=None)
    df_val.columns = ["A", "B", "C"]
    df_test = pd.read_csv('../../data/txt_files/'+ "Test/" + bias_char + "/" + 'Test.csv', delimiter=',', encoding='utf-8',header=None)
    df_test.columns = ["A", "B", "C"]
    sns.set()
    print (df_train,df_val,df_test)
    count_train = df_train["C"].value_counts(ascending=True)
    count_val = df_val["C"].value_counts(ascending=True)
    count_test = df_test["C"].value_counts(ascending=True)
    print(count_train,count_test,count_val)

    plt.hist(df_train['C'])
    plt.hist(df_val['C'])
    plt.hist(df_test['C'])
    plt.ylabel('Frequency count')
    plt.xlabel('Data');
    plt.title('My histogram')
    plt.show()


if __name__ == '__main__':
    file_path = ('../../data/txt_files/Bias_Study.csv')

    config = load_config("./my_config.yaml")
    print(config)

    matched = ('../../data/txt_files/Bias_Study.csv')
    for i in config["Bias_char"]:

        reed_pand(i)

