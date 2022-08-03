#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Prudhvi Thirumalaraju   Line 3
# Created Date: 07/22/2022
# version ='1.0'
# ---------------------------------------------------------------------------
""" Frequency distribution plots for a given dataset (both chars anf histogram for numerical data) """
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import numpy as np

def read():
    with open("./char.csv", encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            print (row)
def reed_pand():
    # df = pd.read_csv('./Bias EmbryoScope Data 4-29-22.csv', encoding='utf-8')
    # print(df.head(0))
    df = pd.read_csv('./char.csv', encoding='utf-8')
    sns.set()
    ncols = len(df.columns)
    print(ncols)
    fig, axes = plt.subplots(2,3)

    for x,ax in (zip(df.head(0), axes.flatten())):
       print (x)
       # df[x].value_counts().plot(kind="hist")
       # sns.set(rc={"figure.figsize": (16, 10)})
       s=sns.countplot(x=df[x],orient='v',ax=ax)
       s.set_xticklabels(s.get_xticklabels(), rotation=45,ha="right",fontsize=15)
       fig=s.get_figure()

       # print(f",./{x}.png")
       # fig.savefig(f"./{x}.png")
       # plt.savefig(f"./{x}.png", bbox_inches='tight')
       # plt.show()
    # p = df["Sperm Source Race"].value_counts().plot(kind='bar')
    #plt.hist(p)
    plt.subplots_adjust(hspace=0.1)
    plt.tight_layout()
    plt.show()
def reed_hist():
    # df = pd.read_csv('./Bias EmbryoScope Data 4-29-22.csv', encoding='utf-8')
    # print(df.head(0))
    df = pd.read_csv('./hist.csv', encoding='utf-8')
    sns.set()
    ncols = len(df.columns)
    print(ncols)
    fig, axes = plt.subplots(3,3)

    for x,ax in (zip(df.head(0), axes.flatten())):
       print (x)
       # df[x].value_counts().plot(kind="hist")
       # sns.set(rc={"figure.figsize": (16, 10)})
       s=sns.histplot(x=df[x],ax=ax, bins=20)
       # s.set_xticklabels(s.get_xticklabels(), rotation=45,ha="right",fontsize=15)
       fig=s.get_figure()
       # print(f",./{x}.png")
       # fig.savefig(f"./{x}.png")
       # plt.savefig(f"./{x}.png", bbox_inches='tight')
       # plt.show()
    # p = df["Sperm Source Race"].value_counts().plot(kind='bar')
    #plt.hist(p)
    plt.subplots_adjust(hspace=0.1)
    plt.tight_layout()
    plt.show()




def get_implantation(file_path:str,bias_char:str):

    df = pd.read_csv(file_path, encoding='utf-8')
    ncols = len(df.columns)
    df1 = df[df["Known Outcome? 1=Yes, 0=No"] == 1]
    print(df1)
    txt_implantation = df1[["EmbryoScope Image ID", "PREG" , bias_char]]
    print (txt_implantation)
    x = txt_implantation[bias_char].value_counts(ascending=True)




    a = x.values.tolist()
    b = x.index.values.tolist()

    arr_count = (np.vstack((a, b)).T)
    num_cat = len (arr_count)
    # print (arr_count)
    print ("Number of catogories = " + str (num_cat))

    Test_set_size_min = 70
    Test_set_size_max = 300

    Per_cat= Test_set_size_min//num_cat

    print (Per_cat)

    dfb =txt_implantation
    for i ,j in arr_count:
        if int(i) >= 40 :
            print (i,j)
            break
        else:
            y = j
            dfb = dfb.drop(dfb[dfb[bias_char] == y].index)


    x = dfb[bias_char].value_counts(ascending=True)
    print (x)
    return dfb

def make_txt_type_1(,bias_char):

    print (dfb)

    x = dfb[bias_char].value_counts(ascending=True)
    a = x.values.tolist()
    b = x.index.values.tolist()

    print (a)
    print (b)
    root = "../../data/raw/"
    for i in b:
        temp = dfb[(dfb[bias_char] == i)]
        xx = temp[bias_char].value_counts(ascending=True)
        shuffled = temp.sample(frac=1).reset_index()
        print (len(shuffled.index))

        file_name = shuffled.values.tolist()
        Train_list = shuffled.values.tolist() [0:int(0.7*len(shuffled.index))]
        Val_list = shuffled.values.tolist() [int(0.7*len(shuffled.index)):int(0.9*len(shuffled.index))]
        Test_list = shuffled.values.tolist() [int(0.9*len(shuffled.index)):int(len(shuffled.index))]

        for f in Train_list:
            print (f)

            with open('../../data/txt_files/' + bias_char+ 'Train' + ".txt", 'a') as the_file:
                  the_file.write(root + str(f[1])+ " " + str(f[2]) + " " + '\n')
        for f in Val_list:
            print (f)                                                        

            with open('../../data/txt_files/' + bias_char+ 'Val' + ".txt", 'a') as the_file:
                  the_file.write(root + str(f[1])+ " " + str(f[2]) + " " + '\n')
        for f in Test_list:
            print (f)

            with open('../../data/txt_files/' + bias_char + 'Test'+ ".txt", 'a') as the_file:
                  the_file.write(root + str(f[1])+ " " + str(f[2]) + " " + '\n')









    # xy = dfb.groupby('Sperm Source Race', group_keys=False).apply(lambda x: x.sample(frac=0.9))
    # print (xy)
    root = "./data/"
    file_name = dfb.values.tolist()
    # print (file_name)
    # for f in file_name:
    #     print (f)
    #     with open('./txt/' + "_train" + ".txt", 'a') as the_file:
    #          the_file.write(root + f[0]+ " " + str(f[1]) + " " + '\n')


    # file_name = df["EmbryoScope Image ID"].tolist()
    # impl = df["EmbryoScope Image ID"].tolist()
    # file_name = txt_implantation.values.tolist()



    # print(file_name)
    # root = './data/'
    # df1 = txt_implantation[df["PREG"] == 0]
    # file_name = df1.values.tolist()
    # for f in file_name:
    #     print (f)
    #     with open('../../data/txt_files/' + bias_char + ".txt", 'a') as the_file:
    #          the_file.write(root + f[0]+ " " + str(f[1]) + " " + '\n')

    # f = open('./txt/' + "_train" + ".txt", 'a')
    # f.writelines(['\n', str(df["EmbryoScope Image ID"]), ' ',str(df["Known Outcome? 1=Yes, 0=No"]) ])
    # f.close()

        # with open('./data/embryo/' + "_train" + ".txt", 'a') as the_file:
            # with open('../data/sd1/val.txt', 'a') as the_file:
            # the_file.write(data_dir_path+img_name+" "+img_name+'\n')
            # the_file.write(source_data_dir_path + classes + "/" + img_name + " " + str(int(classes) - 1) + " " + '0' + '\n')
# Press the green button in the gutter to run the script.

if __name__ == '__main__':

    # f1 = ('../../data/txt_files/Embryoscope Image List 1-27-22.csv')
    # f2 = ('../../data/txt_files/Bias EmbryoScope Data 4-29-22.csv')
    # match(f1,f2)
    # f1 = ('../../data/txt_files/Sperm Source RaceTrain.txt')
    #
    # f2 = ('../../data/txt_files/Sperm Source RaceVal.txt')
    # match(f1,f2)
    matched = ('../../data/txt_files/Bias_Study.csv')
    bias_distrubution = 'Sperm Source Race'
    # match(f1,f2)
    get = get_implantation(matched,bias_distrubution)
    make_txt_type_1(get,bias_distrubution)






    # reed_pand()
    # reed_hist()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
