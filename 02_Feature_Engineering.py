import pandas as pd 
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

"""the first step of feature engineering is to remove low variance features
It is especially useful in datasets with many columns, every data set containing numerical features and projects like Genetic data,Text data,Image data
-->> from sklearn.feature_selection import VarianceThreshold"""

train= pd.read_csv('data/train.csv', header=None)
selector = VarianceThreshold(threshold=0.01)  
train_reduced = selector.fit_transform(train)       
selected_columns = train.columns[selector.get_support()] 
train = train[selected_columns]                        

"""the second step of feature engineering is using a coleration filter for your dataset

"""
corr_matrix = train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column]>0.9)]
train.drop(columns=to_drop, inplace=True)
""" the third step of featutre engineering is 'FEATURECREATİON'
its mean is added new features and columns to the data set that we have Improving the model's performance
the new features can be : mean, std, min, max, sum
                          Differences / Ratios Between Columns
                          Logarithm or Square Root Transformations
                          Binning
                          Interaction Features"""

scaler = StandardScaler()
X_scaled = scaler.fit_transform(train)

pca = PCA(n_components=0.90)  
X_pca = pca.fit_transform(X_scaled)


X_train, X_test, y_train, y_test = train_test_split(X_pca, labels, test_size=0.2)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))