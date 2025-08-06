import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------
# 1. Load Dataset
# ---------------------------
train_data = pd.read_csv('train.csv', header=None)
train_labels = pd.read_csv('trainLabels.csv', header=None)

# ---------------------------
# 2. Remove Low-Variance Features
# ---------------------------
selector = VarianceThreshold(threshold=0.01)
train_reduced = selector.fit_transform(train_data)
selected_columns = train_data.columns[selector.get_support()]
train_data = train_data[selected_columns]

# ---------------------------
# 3. Remove Highly Correlated Features
# ---------------------------
corr_matrix = train_data.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
train_data.drop(columns=to_drop, inplace=True)

# ---------------------------
# 4. Feature Scaling
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_data)

# ---------------------------
# 5. Dimensionality Reduction (PCA)
# ---------------------------
pca = PCA(n_components=0.90)
X_pca = pca.fit_transform(X_scaled)

# ---------------------------
# 6. Train/Validation Split
# ---------------------------
X_train, X_val, y_train, y_val = train_test_split(X_pca, train_labels, test_size=0.2, random_state=42)

# ---------------------------
# 7. Random Forest Classifier
# ---------------------------
model = RandomForestClassifier(random_state=63)
model.fit(X_train, y_train.values.ravel())

# ---------------------------
# 8. Evaluate the Model
# ---------------------------
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"✅ Model Accuracy Rate: {accuracy:.4f}")
