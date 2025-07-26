import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# # print("train_data shape: ", train_data.shape)
# # print("test_data shape: ", test_data.shape)
# # print("train_labels shape: ", train_labels.shape)

# # print(f"train.csv: \n {train_data.head()}")
# # print(f"test.csv: \n {test_data.head()}")
# # print(f"trainLabels.csv: \n {train_labels.head()}")


# print(train_data.head())
# # print(train_data.isnull().sum())

# # print(test_data.isnull().sum())
# # print(train_labels.isnull().sum())



# # train_labels.value_counts().plot(kind='bar')
# # plt.title('label distribution')
# # plt.xlabel('crime type')
# # plt.ylabel('counts')

# # plt.show()


train_data = pd.read_csv('data/train.csv', header=None)
train_labels = pd.read_csv('data/trainLabels.csv', header=None)

# 1. Veriyi eğitim ve test olarak ayırıyoruz
X_train, X_test, y_train, y_test = train_test_split(
    train_data,
    train_labels,
    test_size=0.2,
    random_state=42
)

# 2. Ölçeklendirme yapıyoruz (fit sadece eğitim verisine uygulanır)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Modeli oluşturup eğitiyoruz
my_model = RandomForestClassifier(random_state=63)
my_model.fit(X_train_scaled, y_train.values.ravel())

# 4. Test verisi ile tahmin yapıyoruz
y_pred = my_model.predict(X_test_scaled)

# 5. Doğruluğu hesaplıyoruz
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy rate: {accuracy:.4f}")
