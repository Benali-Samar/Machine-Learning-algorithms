

		# This file is for prediction the rain for 
		#today and tomorrow using a deep learning algorithme "RNA"
		


# librairies importation
import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.stats import norm
from sklearn.neural_network import MLPClassifier


#Dataset importation and data preparation
weather =pd.read_csv('weather.csv')
weather.drop([348] ,axis=0,inplace=True)
weather.drop(['WindGustDir','WindDir9am', 'WindDir3pm'], axis=1, inplace=True)
weather=weather.dropna()

#Data partition for test and train
x=weather.iloc[: , 0:16].values
y= weather.iloc[: , 16].values
y=np.where(y=='Yes' , 1 ,0)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.30, random_state=0)


	#rain today
mlp=MLPClassifier(hidden_layer_sizes=(30,),max_iter=40, alpha=1e-4, solver='sgd', verbose=10, tol=1e-4, 
random_state=1, learning_rate_init=.1)
mlp.fit(x_train,y_train)
print(mlp.score(x_test,y_test))
1-mlp.score(x_test,y_test)



	#rain tomorrow

x1=weather.iloc[: , 0:17].values
x1=np.where(x1=='Yes' , 1 ,0)
y1= weather.iloc[: , 18].values
y1=np.where(y1=='Yes' , 1 ,0)
x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, train_size=0.30, random_state=0)
mlp1=MLPClassifier(hidden_layer_sizes=(30,),max_iter=40, alpha=1e-4, solver='sgd', verbose=10, tol=1e-4, 
random_state=1, learning_rate_init=.1)
mlp1.fit(x_train1,y_train1)
print(mlp1.score(x_test1,y_test1))
1-mlp1.score(x_test1,y_test1)

