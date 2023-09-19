from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# KNeighborsClassifier is a class
# load_iris is a function that returns dataset dictionary
# Model 1 : to predict iris specie using sepal and petal dimensions
# K=8
iris=load_iris()
inpt=iris["data"]
output=iris["target"]
x_train,x_test,y_train,y_test=train_test_split(inpt,output,test_size=0.2)

print(x_train.shape,y_train.shape)


#l=[int(i) for i in input("enter attributes for a single instance(4)").split(" ")]

knn=KNeighborsClassifier(8) #takes K neighbors as arg
knn.fit(x_train,y_train) 
predicted=knn.predict(x_test)

predicted_cats=[ iris["target_names"][i] for i in knn.predict(x_test)]
print(predicted_cats)
#fit() is used to train
#print(f'belongs to : {iris["target_names"][knn.predict([l])]} ')    #predict is used to test the model
print(f"model training accuracy : {knn.score(x_train,y_train)*100}%")
print(f"model testing  accuracy : {accuracy_score(predicted,y_test)*100}%")
