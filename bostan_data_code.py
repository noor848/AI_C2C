
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston

data_b=load_boston()
print(data_b.keys())

data_fram=pd.DataFrame(data=data_b.data,columns=data_b.feature_names)
print(data_fram.head())
data_fram['price']=data_b.target
print(data_fram.head())

print(data_fram.isnull().sum())

corr_matrix=data_fram.corr().round(1)
sns.heatmap(corr_matrix, annot=True)

features=["LSTAT","RM"]
target=data_fram["price"]


for i,col in enumerate(features):
    plt.subplot(1,2,i+1)
    x=data_fram[col]
    y=target
    plt.scatter(x, y, marker="o")
    plt.xlabel(col)
    plt.ylabel("price")
    plt.title(i+1)
    plt.show()