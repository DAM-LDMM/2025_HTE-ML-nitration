import pandas as pd
import numpy as np

# rdkit version : '2024.09.1'
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator



#%% I/O


file        = pd.read_excel("../data/Normalized nitration HTE Data.xlsx") 
y           = file['Ratio P/IS']
output_file = "../data/output_FP.xlsx"

 
#%% Get Substrate SMILES and compute fingerprints

subs         = np.array([Chem.MolToSmiles(Chem.MolFromSmiles(i)) for i in file["Substrate SMILES"]])
subs_mols    = np.array([Chem.MolFromSmiles(i) for i in subs])
unique_subs  = np.unique(subs)

mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)
fps    = np.array([mfpgen.GetFingerprint(i) for i in subs_mols])




#%% 
"""
Example for leave-one-out strategy : 
    get test indices corresponding to substrate 0
"""

test_indices  = np.argwhere(subs==unique_subs[0])
train_indices = np.argwhere(subs!=unique_subs[0])

#%% One hot encoding of nitrating and activation agents 

from sklearn.preprocessing import OneHotEncoder

# Encoding features
x2 = np.array([Chem.MolToSmiles(Chem.MolFromSmiles(i)) for i in file["Nitrating Agent SMILES"]]).reshape(-1, 1)
x3 = []
for i in file["Activation agent SMILES"]:
    if isinstance(i, str) : x3.append(Chem.MolToSmiles(Chem.MolFromSmiles(i)))
    else : x3.append("None")                                       
x3 = np.array(x3).reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False, handle_unknown='error')
X       = np.hstack((encoder.fit_transform(np.hstack((x2, x3))), fps))  # matrix of shape (864*1044), 1024 bits+20 elements for nitrating and activation agents

# Encoding labels as 0 or 1 
Y = []
for value in y:
    if value == 0 : Y.append(0)
    else : Y.append(1)
Y = np.array(Y)




#%% 
"""
Train/val/test split and initialization of different models 
Several tests are made to avergae results
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score


lst_df = []
n_test = 20 
for iteration in range(n_test):
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)  # uncomment this line and comment the following lines for random split instead of leave-one-out. 
    X_train, X_test, Y_train, Y_test = X[train_indices], X[test_indices], Y[train_indices], Y[test_indices]
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[2])
    X_test  = X_test.reshape(X_test.shape[0],X_test.shape[2])
    Y_train = Y_train.reshape(Y_train.shape[0],Y_train.shape[1])
    Y_test  = Y_test.reshape(Y_test.shape[0],Y_test.shape[1])
    
    print("Proportion of succes in train : ", sum(Y_train)/len(Y_train))
    print("Proportion of succes in test : ", sum(Y_test)/len(Y_test)) 
    
    models = {}
    models['Majority Class (dummy)']  = DummyClassifier(strategy="most_frequent")
    models['Random (dummy)']          = DummyClassifier(strategy='stratified')
    models['Random uniform (dummy)']  = DummyClassifier(strategy='uniform')
    models['Logistic Regression']     = LogisticRegression()
    models['Support Vector Machines'] = LinearSVC()
    models['Naive Bayes']             = GaussianNB()
    models['K-Nearest Neighbor']      = KNeighborsClassifier()
    models['AdaBoost']                = AdaBoostClassifier()
    models['Decision Trees']          = DecisionTreeClassifier()
    models['Gradient Boosting']       = GradientBoostingClassifier()
    models['Hist Gradient Boosting']  = HistGradientBoostingClassifier()
    models['Extra Trees']             = ExtraTreesClassifier()
    models['Random Forest']           = RandomForestClassifier()
    
    df = pd.DataFrame(index=models.keys(), columns=['Acc.', 'Prec.', 'Rec.', 'ROC AUC', 'Bal. Acc.'])
    
    for key in models.keys():
        print("Training {} for iteration {}".format(key, iteration))
        models[key].fit(X_train, Y_train) # Fit the classifier
        predictions = models[key].predict(X_test) # Make predictions
        df.loc[key, 'Acc.']  = accuracy_score(predictions, Y_test)
        df.loc[key, 'Prec.'] = precision_score(predictions, Y_test)
        df.loc[key, 'Rec.']  = recall_score(predictions, Y_test)
        df.loc[key, 'ROC AUC']  = roc_auc_score(Y_test, predictions)
        df.loc[key, 'Bal. Acc.']  = balanced_accuracy_score(Y_test, predictions)

    
    lst_df.append(df)

df_average = pd.DataFrame(index=df.index, columns=['Acc.', 'Acc. stdev', 'Prec.', 'Prec. stdev', 'Rec.', 'Rec. stdev', 'ROC AUC', 'ROC AUC stdev', 'Bal. Acc.', 'Bal. Acc. stdev'])
df_average['Acc.']        = np.mean([i['Acc.'].values for i in lst_df], axis=0)
df_average['Prec.']       = np.mean([i['Prec.'].values for i in lst_df], axis=0)
df_average['Rec.']        = np.mean([i['Rec.'].values for i in lst_df], axis=0)
df_average['ROC AUC']     = np.mean([i['ROC AUC'].values for i in lst_df], axis=0)
df_average['Bal. Acc.']   = np.mean([i['Bal. Acc.'].values for i in lst_df], axis=0)


df_average['Acc. stdev']  = [np.std(i) for i in [[np.array([i['Acc.'].values for i in lst_df])[:,j]] for j in range(len(models.keys()))]]
df_average['Prec. stdev'] = [np.std(i) for i in [[np.array([i['Prec.'].values for i in lst_df])[:,j]] for j in range(len(models.keys()))]]
df_average['Rec. stdev']  = [np.std(i) for i in [[np.array([i['Rec.'].values for i in lst_df])[:,j]] for j in range(len(models.keys()))]]
df_average['ROC AUC stdev']    = [np.std(i) for i in [[np.array([i['ROC AUC'].values for i in lst_df])[:,j]] for j in range(len(models.keys()))]]
df_average['Bal. Acc. stdev']  = [np.std(i) for i in [[np.array([i['Bal. Acc.'].values for i in lst_df])[:,j]] for j in range(len(models.keys()))]]
df_average.sort_values('ROC AUC', inplace=True)



df_average.to_excel(output_file)

