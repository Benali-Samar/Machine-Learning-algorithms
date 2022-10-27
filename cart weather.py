
			#Classification and regression trees "CART" algorithme 
			#implementation for rain prediction
			
			
#librairies importation			 
from sklearn import tree
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#dataset importation
weather =pd.read_csv('weather.csv')


#data preparation
weather.drop([348] ,axis=0,inplace=True)
weather.drop(['WindGustDir','WindDir9am', 'WindDir3pm'], axis=1, inplace=True)
weather=weather.dropna()


#data partition for train and test
x=weather.iloc[: , 0:16].values
y= weather.iloc[: , 16].values
y=np.where(y=='Yes' , 1 ,0)
	
	#rain today
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.30, random_state=0)
arbre=tree.DecisionTreeClassifier() 
arbre.fit(x_train,y_train)
arbre.score(x_test,y_test)
1-arbre.score(x_test,y_test)
ypredicted=arbre.predict(x_test)
cm=confusion_matrix(y_test,ypredicted)
cm


	#rain tomorrow
x1=weather.iloc[: , 0:17].values
x1=np.where(x1=='Yes' , 1 ,0)
y1= weather.iloc[: , 18].values
y1=np.where(y1=='Yes' , 1 ,0)
x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, train_size=0.30, random_state=0)
arbre1=tree.DecisionTreeClassifier() 
arbre1.fit(x_train1,y_train1)
arbre1.score(x_test1,y_test1)
1-arbre1.score(x_test1,y_test1)
ypredicted1=arbre1.predict(x_test1)
cm1=confusion_matrix(y_test1,ypredicted1)
cm1


