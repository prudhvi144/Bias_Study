#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Prudhvi Thirumalaraju   Line 3
# Created Date: 07/29/2022
# version ='1.0'
# ---------------------------------------------------------------------------
""" mathch the data and filter it -> returns new csv files """
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import yaml
import numpy as np

def filter(file_path:str,bias_char:str):

    df = pd.read_csv(file_path, encoding='utf-8')
    df1 = df[df["Known Outcome? 1=Yes, 0=No"] == 1]
    trim = df1[["EmbryoScope Image ID", "PREG", bias_char]]
    categories = trim[bias_char].value_counts(ascending=True)
    a = categories.values.tolist()
    b = categories.index.values.tolist()
    arr_count = (np.vstack((a, b)).T)
    num_cat = len(arr_count)
    print("Number of catogories = " + str(num_cat))

    arr_count = (np.vstack((a, b)).T)
    num_cat = len (arr_count)
    # print (arr_count)
    print ("Number of catogories = " + str (num_cat))

    Test_set_size_min = 200
    Test_set_size_max = 300

    Per_cat= Test_set_size_min//num_cat
    for i, j in arr_count:
        if int(i) >= Per_cat:
            print(i, j)
            break
        else:
            y = j
            trim = trim.drop(trim[trim[bias_char] == y].index)
    x = trim[bias_char].value_counts(ascending=True)
    print(x)
    return trim

def stratfied(dfb,bias_char:str):
    txt = "stratfied"
    x = dfb[bias_char].value_counts(ascending=True)
    a = x.values.tolist()
    b = x.index.values.tolist()
    root = "../../data/raw/"
    parent_dir = "../../data/txt_files/"+ txt+ "/"
    test_dir = "../../data/txt_files/"+ txt+ "/"+"Test/"
    path = os.path.join(parent_dir, bias_char)
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(test_dir, bias_char)
    if not os.path.exists(path):
        os.makedirs(path)

    for i in b:
        temp = dfb[(dfb[bias_char] == i)]
        xx = temp[bias_char].value_counts(ascending=True)
        shuffled = temp.sample(frac=1,random_state=0).reset_index()
        print(len(shuffled.index))

        file_name = shuffled.values.tolist()
        Train_list = shuffled.values.tolist()[0:int(0.7 * len(shuffled.index))]
        Val_list = shuffled.values.tolist()[int(0.7 * len(shuffled.index)):int(0.9 * len(shuffled.index))]
        Test_list = shuffled.values.tolist()[int(0.9 * len(shuffled.index)):int(len(shuffled.index))]

        for f in Train_list:
            print(f)

            with open(parent_dir+bias_char+"/"  + 'Train' + ".txt", 'a') as the_file:
                the_file.write(root + str(f[1]) + " " + str(f[2]) + " " + '\n')
            with open(parent_dir+bias_char+"/" + 'Train.csv', 'a', newline='',) as file:
                csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow([f[1],f[2],f[3]])

        for f in Val_list:
            print(f)

            with open(parent_dir+bias_char+"/" + 'Val' + ".txt", 'a') as the_file:
                the_file.write(root + str(f[1]) + " " + str(f[2]) + " " + '\n')
            with open(parent_dir+bias_char+"/" + 'Val.csv', 'a', newline='',) as file:
                csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow([f[1],f[2],f[3]])
        for f in Test_list:
            print(f)
            test_dir = "../../data/txt_files/"+txt+'/'+bias_char+"/Test/"
            path = os.path.join(test_dir,str(f[3]))

            if not os.path.exists(path):
                os.makedirs(path)
            with open('../../data/txt_files/'+txt+'/'+bias_char+"/"+ 'Test' + ".txt", 'a') as the_file:
                the_file.write(root + str(f[1]) + " " + str(f[2]) + " " + '\n')
            with open('../../data/txt_files/'+txt+"/" + bias_char +"/"+ "Test/"+ str(f[3])+"/" +  'Test' + ".txt", 'a') as the_file:
                the_file.write(root + str(f[1]) + " " + str(f[2]) + " " + '\n')
            with open('../../data/txt_files/'+ txt +"/Test"+"/" + bias_char + "/" + 'Test.csv', 'a', newline='',) as file:
                csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow([f[1],f[2],f[3]])

def balanced(dfb,bias_char:str):
    txt = "balanced"
    z = dfb[bias_char].value_counts(ascending=True)
    a = z.values.tolist()

    g = z.iloc[0].tolist()

    df = dfb.groupby(bias_char, group_keys=False).apply(lambda x: x.sample(g,random_state=1))

    y = df[bias_char].value_counts(ascending=True)
    b = y.index.values.tolist()
    # data_clean = dfb.drop(df1.index)
    # z = data_clean[bias_char].value_counts(ascending=True)

    root = "../../data/raw/"
    parent_dir = "../../data/txt_files/"+ txt+ "/"
    test_dir = "../../data/txt_files/"+ txt+ "/"+"Test/"
    path = os.path.join(parent_dir, bias_char)

    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(test_dir, bias_char)
    if not os.path.exists(path):
        os.makedirs(path)

    for i in b:
        temp = df[(df[bias_char] == i)]
        xx = temp[bias_char].value_counts(ascending=True)
        shuffled = temp.sample(frac=1,random_state=0).reset_index()
        print(len(shuffled.index))

        file_name = shuffled.values.tolist()
        Train_list = shuffled.values.tolist()[0:int(0.7 * len(shuffled.index))]
        Val_list = shuffled.values.tolist()[int(0.7 * len(shuffled.index)):int(0.9 * len(shuffled.index))]
        Test_list = shuffled.values.tolist()[int(0.9 * len(shuffled.index)):int(len(shuffled.index))]

        for f in Train_list:
            print(f)

            with open(parent_dir+bias_char+"/"  + 'Train' + ".txt", 'a') as the_file:
                the_file.write(root + str(f[1]) + " " + str(f[2]) + " " + '\n')
            with open(parent_dir+bias_char+"/" + 'Train.csv', 'a', newline='',) as file:
                csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow([f[1],f[2],f[3]])

        for f in Val_list:
            print(f)

            with open(parent_dir+bias_char+"/" + 'Val' + ".txt", 'a') as the_file:
                the_file.write(root + str(f[1]) + " " + str(f[2]) + " " + '\n')
            with open(parent_dir+bias_char+"/" + 'Val.csv', 'a', newline='',) as file:
                csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow([f[1],f[2],f[3]])
        for f in Test_list:
            print(f)
            test_dir = "../../data/txt_files/"+txt+'/'+bias_char+"/Test/"
            path = os.path.join(test_dir,str(f[3]))

            if not os.path.exists(path):
                os.makedirs(path)
            with open('../../data/txt_files/'+txt+'/'+bias_char+"/"+ 'Test' + ".txt", 'a') as the_file:
                the_file.write(root + str(f[1]) + " " + str(f[2]) + " " + '\n')
            with open('../../data/txt_files/'+txt+"/" + bias_char +"/"+ "Test/"+ str(f[3])+"/" +  'Test' + ".txt", 'a') as the_file:
                the_file.write(root + str(f[1]) + " " + str(f[2]) + " " + '\n')
            with open('../../data/txt_files/'+ txt +"/Test"+"/" + bias_char + "/" + 'Test.csv', 'a', newline='',) as file:
                csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow([f[1],f[2],f[3]])




    pass
def residual(dfb,bias_char:str):
    txt = "residual"
    x = dfb[bias_char].value_counts(ascending=True)

    g = x.iloc[0].tolist()

    df = dfb.groupby(bias_char, group_keys=False).apply(lambda x: x.sample(g,random_state=1))
    y = df[bias_char].value_counts(ascending=True)
    cc = y.index.values.tolist()
    df1 = dfb.drop(df.index)
    z = df1[bias_char].value_counts(ascending=True)
    bb = z.index.values.tolist()
    root = "../../data/raw/"
    parent_dir = "../../data/txt_files/" + txt + "/"
    test_dir = "../../data/txt_files/" + txt + "/" + "Test/"
    path = os.path.join(parent_dir, bias_char)

    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(test_dir, bias_char)
    if not os.path.exists(path):
        os.makedirs(path)
    for i in cc:
        temp = df[(df[bias_char] == i)]
        xx = temp[bias_char].value_counts(ascending=True)
        shuffled = temp.sample(frac=1, random_state=0).reset_index()
        print(len(shuffled.index))

        file_name = shuffled.values.tolist()
        Train_list = shuffled.values.tolist()[0:int(0.7 * len(shuffled.index))]
        Val_list = shuffled.values.tolist()[int(0.7 * len(shuffled.index)):int(0.9 * len(shuffled.index))]
        Test_list = shuffled.values.tolist()[int(0.9 * len(shuffled.index)):int(len(shuffled.index))]

        for f in Train_list:
            print(f)

            with open(parent_dir + bias_char + "/" + 'Train' + ".txt", 'a') as the_file:
                the_file.write(root + str(f[1]) + " " + str(f[2]) + " " + '\n')
            with open(parent_dir + bias_char + "/" + 'Train.csv', 'a', newline='', ) as file:
                csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow([f[1], f[2], f[3]])

        for f in Val_list:
            print(f)

            with open(parent_dir + bias_char + "/" + 'Val' + ".txt", 'a') as the_file:
                the_file.write(root + str(f[1]) + " " + str(f[2]) + " " + '\n')
            with open(parent_dir + bias_char + "/" + 'Val.csv', 'a', newline='', ) as file:
                csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow([f[1], f[2], f[3]])
        for f in Test_list:
            print(f)
            test_dir = "../../data/txt_files/" + txt + '/' + bias_char + "/Test/"
            path = os.path.join(test_dir, str(f[3]))

            if not os.path.exists(path):
                os.makedirs(path)
            with open('../../data/txt_files/' + txt + '/' + bias_char + "/" + 'Test' + ".txt", 'a') as the_file:
                the_file.write(root + str(f[1]) + " " + str(f[2]) + " " + '\n')
            with open(
                    '../../data/txt_files/' + txt + "/" + bias_char + "/" + "Test/" + str(f[3]) + "/" + 'Test' + ".txt",
                    'a') as the_file:
                the_file.write(root + str(f[1]) + " " + str(f[2]) + " " + '\n')
            with open('../../data/txt_files/' + txt + "/Test" + "/" + bias_char + "/" + 'Test.csv', 'a',
                      newline='', ) as file:
                csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow([f[1], f[2], f[3]])
    for i in bb:
        temp = df[(df[bias_char] == i)]
        xx = temp[bias_char].value_counts(ascending=True)
        shuffled = temp.sample(frac=1, random_state=0).reset_index()
        print(len(shuffled.index))

        file_name = shuffled.values.tolist()
        Train_list = shuffled.values.tolist()[0:int(0.9 * len(shuffled.index))]
        Val_list = shuffled.values.tolist()[int(0.9* len(shuffled.index)):int(1 * len(shuffled.index))]


        for f in Train_list:
            print(f)

            with open(parent_dir + bias_char + "/" + 'Train' + ".txt", 'a') as the_file:
                the_file.write(root + str(f[1]) + " " + str(f[2]) + " " + '\n')
            with open(parent_dir + bias_char + "/" + 'Train.csv', 'a', newline='', ) as file:
                csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow([f[1], f[2], f[3]])

        for f in Val_list:
            print(f)

            with open(parent_dir + bias_char + "/" + 'Val' + ".txt", 'a') as the_file:
                the_file.write(root + str(f[1]) + " " + str(f[2]) + " " + '\n')
            with open(parent_dir + bias_char + "/" + 'Val.csv', 'a', newline='', ) as file:
                csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow([f[1], f[2], f[3]])




    pass
CONFIG_PATH = "./"
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


if __name__ == '__main__':
    file_path = ('../../data/txt_files/Bias_Study.csv')

    config = load_config("./my_config.yaml")
    print(config)


    matched = ('../../data/txt_files/Bias_Study.csv')
    for i in config["Bias_char"]:
        dfb = filter(file_path,i)
        # stratfied(dfb,i)
        # balanced(dfb,i)
        residual(dfb,i)

    #



