import pandas as pd 
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import VarianceThreshold


"""the first step of feature engineering is to remove low variance features
It is especially useful in datasets with many columns, every data set containing numerical features and projects like Genetic data,Text data,Image data
-->> from sklearn.feature_selection import VarianceThreshold"""

train= pd.read_csv('data/train.csv', header=None)
selector = VarianceThreshold(threshold=0.01)  
train_reduced = selector.fit_transform(train)       
selected_columns = train.columns[selector.get_support()] 
train = train[selected_columns]                        


corr_matrix = train.corr().abs()
print(corr_matrix)