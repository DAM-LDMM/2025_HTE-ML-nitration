import pandas as pd
import numpy as np
import matplotlib as mpl
FONT = {'size' : 12}
mpl.rc('font', **FONT)
import matplotlib.pyplot as plt






#%%

wd     = "../data/" 
algo_order = ['Gradient Boosting', 'Naive Bayes', 'AdaBoost', 'Hist Gradient Boosting', 'Logistic Regression', 'Support Vector Machines', 'Extra Trees', 'Random Forest', 'Decision Trees', 'K-Nearest Neighbor', 'Majority Class (dummy)', 'Random (dummy)', 'Random uniform (dummy)']
fp512  = pd.read_excel(wd+"MorganFP-2-512.xlsx", index_col=0)
fp1024 = pd.read_excel(wd+"MorganFP-2-1024.xlsx", index_col=0) 
fp2048 = pd.read_excel(wd+"MorganFP-2-2048.xlsx", index_col=0) 
rd125  = pd.read_excel(wd+"rdkit-125.xlsx", index_col=0) 
rd210  = pd.read_excel(wd+"rdkit-210.xlsx", index_col=0) 


df = pd.DataFrame()
df['MorganFP-2-512']  = fp512['Bal. Acc.'] 
df['MorganFP-2-1024'] = fp1024['Bal. Acc.']
df['MorganFP-2-2048'] = fp2048['Bal. Acc.']
df['rdkit-125']       = rd125['Bal. Acc.']
df['rdkit-210']       = rd210['Bal. Acc.']


err = pd.DataFrame()
err['MorganFP-2-512 stdev']  = fp512['Bal. Acc. stdev']
err['MorganFP-2-1024 stdev'] = fp1024['Bal. Acc. stdev']
err['MorganFP-2-2048 stdev'] = fp2048['Bal. Acc. stdev']
err['rdkit-125 stdev']       = rd125['Bal. Acc. stdev']
err['rdkit-210 stdev']       = rd210['Bal. Acc. stdev']


round_to = 2
annot    = pd.DataFrame()
annot['MorganFP-2-512']  = np.round(df['MorganFP-2-512'],round_to).astype(str)+'±'+np.round(err['MorganFP-2-512 stdev'], round_to).astype(str)
annot['MorganFP-2-1024'] = np.round(df['MorganFP-2-1024'],round_to).astype(str)+'±'+np.round(err['MorganFP-2-1024 stdev'],round_to).astype(str)
annot['MorganFP-2-2048'] = np.round(df['MorganFP-2-2048'],round_to).astype(str)+'±'+np.round(err['MorganFP-2-2048 stdev'],round_to).astype(str)
annot['rdkit-125']       = np.round(df['rdkit-125'],round_to).astype(str)+'±'+np.round(err['rdkit-125 stdev'],round_to).astype(str)
annot['rdkit-210']       = np.round(df['rdkit-210'],round_to).astype(str)+'±'+np.round(err['rdkit-210 stdev'],round_to).astype(str)





import seaborn as sns 
plt.figure(figsize=(12,10))
sns.heatmap(df, annot=annot, cmap='PuBu', fmt='', cbar_kws={'label': 'Balanced accuracy'}, alpha=0.7)
plt.xticks(rotation = 25, ha='right', rotation_mode='anchor')
plt.show()       
# plt.savefig("../figs/fig4a.svg", format='svg', bbox_inches="tight")
      
