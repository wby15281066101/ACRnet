# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 20:44:52 2020

@author: wby
"""

import pandas as pd
import os
import shutil

f=open("qwe.csv","rb")
#f=open("ants-bees.csv","rb")
list=pd.read_csv(f)
list["FILE_ID_PNG"]=".png"
list["FILE_ID1"]=list["FILE_ID"]+list["FILE_ID_PNG"]


for h in range(7):
    try:
        os.mkdir(str(h))
    except OSError:
        pass
    continue


for i in range(7):
    try:
        listnew=list[list["CATEGORY_ID"]==i]
#        print(listnew)
        l=listnew["FILE_ID1"].tolist() 
#        print(l,len(l))
        j=str(i)
#        print(j)
        
        for each in l:
            try:
                shutil.move(each,j)
            except OSError:
                pass
            continue     
    
    except OSError:
          pass
    continue
print('finish!')


