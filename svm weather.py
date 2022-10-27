


			#this fileis for rain prediction for today and tomorrow using
			# a classifier algorithme " Support Vector Machine : SVM"
			

	# Biblio importation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.model_selection import train_test_split



	#dataset importation

weather =pd.read_csv('weather.csv')
weather


	#traitement de données

weather.drop([348] ,axis=0,inplace=True)
weather.drop(['WindGustDir','WindDir9am', 'WindDir3pm'], axis=1, inplace=True)
weather=weather.dropna()
weather.shape


	#Data partition
x=weather.iloc[: , 0:16].values
y= weather.iloc[: , 16].values
y=np.where(y=='Yes' , 1 ,0)

		#rain today
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.30, random_state=0)
clf= svm.LinearSVC()
print clf.fit(x,y)
print clf.score(x,y)
print 1-clf.score(x,y)
		#rain today prediction
ypredicted=clf.predict(x_test)
cm=confusion_matrix(y_test,ypredicted)
print cm


		#rain tomorrow
x1=weather.iloc[: , 0:17].values
x1=np.where(x1=='Yes' , 1 ,0)
y1= weather.iloc[: , 18].values
y1=np.where(y1=='Yes' , 1 ,0)


		#rain tomorrow prediction
x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, train_size=0.30, random_state=0)
print ('shape for training data',x_train1.shape ,y_train1.shape,x_test1.shape,y_test1.shape)
clf1= svm.LinearSVC()
clf1.fit(x1,y1)
clf1.score(x1,y1)
1-clf1.score(x1,y1)
ypredicted1=clf1.predict(x_test1)
cm1=confusion_matrix(y_test1,ypredicted1)
cm1






