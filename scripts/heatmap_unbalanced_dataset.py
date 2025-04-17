import pandas as pd
import numpy as np
import matplotlib as mpl
FONT = {'size' : 12}
mpl.rc('font', **FONT)
import matplotlib.pyplot as plt


from rdkit import Chem
from rdkit.Chem import Draw, rdFingerprintGenerator
from rdkit.Chem.Draw import IPythonConsole

import glob


#%%



wd              = "../data/"  
files = glob.glob(wd+"test_unbalanced_0.*-0.*.xlsx")
prop_success_train, prop_success_test, balacc = [], [], []


for file in files : 
    splitted_name      = file.split('/')[-1].split('test_unbalanced_')[-1].split('.xlsx')[0].split('-')
    prop_success_train.append(float(splitted_name[0]))
    prop_success_test.append(float(splitted_name[1]))
    df = pd.read_excel(file)
    balacc.append(df.iloc[-1]['Bal. Acc.'])



df = pd.DataFrame()
df["Proportion of sucessfull experiments in testset"]  = pd.Series(prop_success_test)
df["Proportion of sucessfull experiments in trainset"] = pd.Series(prop_success_train)
df["Balanced accuracy of best model"] = pd.Series(balacc)

    


#%%



 
import seaborn as sns 
sns.set(font_scale=1.5)
plt.figure(figsize=(15,10))
sns.heatmap(df.pivot(columns='Proportion of sucessfull experiments in trainset', index='Proportion of sucessfull experiments in testset',\
                     values='Balanced accuracy of best model'), annot=True, cmap='PuBu', fmt='.3f', cbar_kws={'label': 'Accuracy of best model'}, alpha=0.7)
plt.show()       
# plt.savefig("../figs/heatmap_balanced_accuracy.svg", format='svg', bbox_inches="tight")
      
