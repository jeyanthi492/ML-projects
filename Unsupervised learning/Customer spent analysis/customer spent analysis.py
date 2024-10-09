import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('customer_data.csv')

print(dataset.shape)
print(dataset.describe())
print(dataset.head(5))

# Segregating and Zipping Dataset

income = dataset['income'].values
spend = dataset['spending'].values
x = np.array(list(zip(income,spend)))
print(x)

from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
  km=KMeans(n_clusters=i,random_state=0)
  km.fit(x)
  wcss.append(km.inertia_)
plt.plot(range(1,11),wcss, color = "red", marker="8")
plt.title('Optimal K value')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()

model = KMeans(n_clusters=4, random_state=0)
y_means = model.fit_predict(x)

"""Visualizing the clusters for K=4

Cluster 1: Customers with medium income and low spend

Cluster 2: Customers with high income and medium to high spend

Cluster 3: Customer with low income

Cluster 4 : Customers with medium income but high spend
"""

plt.scatter(x[y_means==0,0],x[y_means==0,1],s=50,c='brown',label='1')
plt.scatter(x[y_means==1,0],x[y_means==1,1],s=50,c='blue',label='2')
plt.scatter(x[y_means==2,0],x[y_means==2,1],s=50,c='green',label='3')
plt.scatter(x[y_means==3,0],x[y_means==3,1],s=50,c='cyan',label='4')
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],s=100,marker ='s',c='red',label='Centroids')
plt.title('Customer spent analysis')
plt.xlabel('Income')
plt.ylabel('Spent')
plt.legend()
plt.show()