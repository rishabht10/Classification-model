from sklearn.neighbors import KNeighborsClassifier

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Model 2: wine category predictor 
# K=10
# there are 3 categories class_1, class_2, class_3
# classification based on 13 attributes

wine_dataset=load_wine()
# print(wine_dataset["data"])
# print(wine_dataset["target"])

input1=wine_dataset["data"]
output1=wine_dataset["target"]

x_train,x_test,y_train,y_test=train_test_split(input1,output1,test_size=0.2)

knn1=KNeighborsClassifier(10,algorithm="auto",weights="distance")
knn1.fit(x_train,y_train)

#l=[int(i) for i in input("enter attributes for a single instance(13)").split(" ")]

#catg=wine_dataset["target_names"][knn1.predict([l])]
#print(f"belongs to : {catg}")
#knn.score(in,out) : training accuracy
predicted=knn1.predict(x_test)
predicted_cats=[wine_dataset["target_names"][i] for i in predicted]

print(predicted_cats)
print(f"model training accuracy : {knn1.score(x_train,y_train)*100}%")
#accuracy_score() takes observed output (predicted) and actual output as args
print(f"model testing  accuracy : {accuracy_score(predicted,y_test)*100}%")
