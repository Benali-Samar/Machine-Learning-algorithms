
			# Kmeans clustering algorithme implimentation for rain prediction


#librairies importation
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


#data preparation
weather =pd.read_csv('weather.csv')
weather.drop([348] ,axis=0,inplace=True)
weather.drop(['WindGustDir','WindDir9am', 'WindDir3pm'], axis=1, inplace=True)
weather=weather.dropna()

#data partition for train and test
x=weather.iloc[: , 0:16].values
y= weather.iloc[: , 16].values
y=np.where(y=='Yes' , 1 ,0)

	#rain today
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.80, random_state=0)
kmeans = KMeans(n_clusters=3)
kmeans.fit(x_train,y_train)
centers=kmeans.cluster_centers_
centers
x, y = make_blobs(n_samples=300, centers=centers, cluster_std=0.60, random_state=0)
plt.scatter(x[:,0],x[:,1])
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(x)
plt.scatter(x[:,0], x[:,1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.show()


	#rain tomorrow
x1=weather.iloc[: , 0:17].values
x1=np.where(x1=='Yes' , 1 ,0)
y1= weather.iloc[: , 18].values
y1=np.where(y1=='Yes' , 1 ,0)
x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, train_size=0.30, random_state=0)
kmeans1 = KMeans(n_clusters=3)
kmeans1.fit(x_train,y_train)
centers1=kmeans1.cluster_centers_
centers1
xa, ya = make_blobs(n_samples=300, centers=centers1, cluster_std=0.60, random_state=0)
plt.scatter(xa[:,0],xa[:,1])
kmeans1 = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans1.fit_predict(xa)
plt.scatter(xa[:,0], xa[:,1])
plt.scatter(kmeans1.cluster_centers_[:, 0], kmeans1.cluster_centers_[:, 1], s=300, c='red')
plt.show()




