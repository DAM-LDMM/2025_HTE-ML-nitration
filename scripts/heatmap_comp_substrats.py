import pandas as pd
import matplotlib as mpl
FONT = {'size' : 12}
mpl.rc('font', **FONT)
import matplotlib.pyplot as plt





#%%



wd              = "../data/"
benzofuran      = pd.read_excel(wd+"output_FP_loo_benzofuran.xlsx", index_col=0)
benzofuran_cooh = pd.read_excel(wd+"output_FP_loo_benzofuran_cooh.xlsx", index_col=0)
benzofuran_br   = pd.read_excel(wd+"output_FP_loo_benzofuran_br.xlsx", index_col=0)
naphta_br       = pd.read_excel(wd+"output_FP_loo_naphta_br.xlsx", index_col=0)
pyri_br         = pd.read_excel(wd+"output_FP_loo_pyri_br.xlsx", index_col=0)
naphta          = pd.read_excel(wd+"output_FP_loo_naphta.xlsx", index_col=0)
naphta_cooh     = pd.read_excel(wd+"output_FP_loo_naphta_cooh.xlsx", index_col=0)
pyri_cooh       = pd.read_excel(wd+"output_FP_loo_pyri_cooh.xlsx", index_col=0)
pyri            = pd.read_excel(wd+"output_FP_loo_pyri.xlsx", index_col=0)




#%%


df = pd.DataFrame()
df['1-bromonaphthalene']           = naphta_br['Bal. Acc.']
df['1-naphtoic acid']              = naphta_cooh['Bal. Acc.']
df['Napthalene']                   = naphta['Bal. Acc.']
df['2-bromobenzofuran']            = benzofuran_br['Bal. Acc.']
df['Benzofuran-2-carboxylic acid'] = benzofuran_cooh['Bal. Acc.']
df['Benzofuran']                   = benzofuran['Bal. Acc.']
df['2-bromopyridine']              = pyri_br['Bal. Acc.']
df['Picolinic acid']               = pyri_cooh['Bal. Acc.']
df['Pyridine']                     = pyri['Bal. Acc.']




 
import seaborn as sns 
plt.figure(figsize=(12,10))
sns.heatmap(df, annot=True, cmap='PuBu', fmt='.3f', cbar_kws={'label': 'Balanced accuracy'}, alpha=0.7)
plt.xticks(rotation = 25, ha='right', rotation_mode='anchor')
plt.show()       
# plt.savefig("../figs/fig5a.svg", format='svg', bbox_inches="tight")
      
