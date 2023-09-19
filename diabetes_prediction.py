#Naive Bayes Classifier Model to predict if a patient
#NB Algo uses Naive Bayes probability theorem to make prediction
import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# reading kaggle dataset
data=pd.read_csv("/kaggle/input/naive-bayes-classification-data/Naive-Bayes-Classification-Data.csv")
print(data.info())

x=data.iloc[:,:2] #extracting input data from dataset
y=data["diabetes"] #extracting output data from dataset

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#Naive Bayes algo
nb=GaussianNB()
nb.fit(x_train,y_train)   #training

predicted=nb.predict(x_test) #predicted output values
trainAcc=nb.score(x_train,y_train)
testAcc=accuracy_score(predicted,y_test)

print(f"training acc :{trainAcc*100}%")
print(f"testing acc :{testAcc*100}%")

# plotting the test patients vs test output 
#blue for diabetic red for non diabetic
patients=range(199) #number of patients for testing
colors = ['red' if val == 0 else 'blue' for val in predicted]
plt.scatter(patients,predicted,c= colors)
plt.xlabel("patients")
plt.ylabel("output")
plt.title("Result analysis")
plt.show()
