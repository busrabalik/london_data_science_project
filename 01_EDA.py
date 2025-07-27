import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



train_data = pd.read_csv('data/train.csv', header=None)
train_labels = pd.read_csv('data/trainLabels.csv', header=None)


# print("train_data shape: ", train_data.shape)
# print("train_labels shape: ", train_labels.shape)

# print(f"train.csv: \n {train_data.head()}")
# print(f"trainLabels.csv: \n {train_labels.head()}")


# print(train_data.head())
# print(train_data.isnull().sum())

# print(train_data.isnull().sum())
# print(train_labels.isnull().sum())



# train_labels.value_counts().plot(kind='bar')
# plt.title('label distribution')
# plt.xlabel('crime type')
# plt.ylabel('counts')

# plt.show()


# correlation_matrix = train_data.corr()
# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
# plt.title("corelation Matrix")
# plt.show()

correlation_with_target = train_data.corrwith(train_labels[0])
print(correlation_with_target.sort_values(ascending=False))