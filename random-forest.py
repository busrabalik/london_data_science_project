import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


train_data = pd.read_csv('data/train.csv', header=None)
train_labels = pd.read_csv('data/trainLabels.csv', header=None)


X_train, X_test, y_train, y_test = train_test_split(
    train_data,
    train_labels,
    test_size=0.2,
    random_state=42
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


my_model = RandomForestClassifier(random_state=63)
my_model.fit(X_train_scaled, y_train.values.ravel())

y_pred = my_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy rate: {accuracy:.4f}")


# output  ->   Model accuracy rate: 0.8900