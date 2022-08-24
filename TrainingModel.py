# %%
#!/usr/bin/env python
import pandas as pd
from sklearn.svm import OneClassSVM
import numpy as np
import os
import time
from sklearn.preprocessing import StandardScaler
#from sklearn import svm
from numpy import where
import joblib
current_dir = os.path.dirname(os.path.abspath(__file__))
# %%
df = pd.read_csv(current_dir + '/Data/prescription.csv',encoding="cp1252", header=0) #Importing Data
df = df[['drugno', 'dose', 'age', 'weight']]  #Selecting Columns
#df.head()
svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.01, verbose=True) # Setting OneClassSVM model
print(svm)
svm.fit(df)  #Traing model
filename = 'finalized_model.sav' #Model file name
joblib.dump(svm, filename) #Saving trained Model


#data = [[5, '1000',76,'65']]
#df_unseen = pd.DataFrame(data,columns=['drugno','dose','age','weight'],dtype=float)
#pred = svm.predict(df_unseen)

#anom_index = where(pred==-1)
#values = df_unseen[anom_index]

# anom = setup(data = df, silent = True)
# models()
# anom_model = create_model(model = 'iforest', fraction = 0.05)
# save_model(model = anom_model, model_name = 'iforest_model')
# loaded_model = load_model('iforest_model')
# df_unseen = ['5','1000','88','102']
# loaded_model.predict(df_unseen)