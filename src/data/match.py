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
import numpy as np

def match( root : str, root_file :str)->int:

    '''
    Get two root folder pathes to compare and check if the files are present in the dataset.
    if there are any mis matches create a new excel sheet without the missing files/images.
    '''

    df1 = pd.read_excel(root)
    df2 = pd.read_csv(root_file)
    # print(df.head())
    i =0

    length = len(df2["EmbryoScope Image ID"])
    for y in df2["EmbryoScope Image ID"]:
        # print (y)
        flag = 0
        for x in df1.Name:
            if (y==x):
               i+=1
               flag = 1
        if flag==0:
            dfb = df2.drop(df2[df2["EmbryoScope Image ID"] == y].index)

    print(len(df2))
    dfb.to_csv('../../data/txt_files/Bias_Study.csv')

    print(length)
    print(f"{i}")
    return length


if __name__ == '__main__':

    f1 = ('../../data/txt_files/Embryoscope Image List 1-27-22.csv')
    f2 = ('../../data/txt_files/Bias EmbryoScope Data 4-29-22.csv')
    match(f1,f2)

