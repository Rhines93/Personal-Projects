import pandas as pd
import numpy as np
import sklearn as sk

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

TD3 = pd.read_csv('Datasets/TrainData3.txt', sep="\t", header=None)
TL3 = pd.read_csv('Datasets/TrainLabel3.txt', sep="\t", header=None)
TestD3 = pd.read_csv('Datasets/TestData3.txt', sep="\t", header=None)

TD3.fillna(TD3.median(), inplace=True)
TestD3.fillna(TestD3.median(), inplace=True)
TD3cat = pd.concat([TD3, TL3], axis=1)
print(TD3.shape)
print(TL3.shape)
print(TestD3.shape)

x3 = TD3cat.iloc[:,0:]
y3 = TD3cat.iloc[:,13]

x_real3 = TD3.iloc[:,0:]
y_real3 = TL3.iloc[:,0]

X_train3, X_test3, y_train3, y_test3 = train_test_split(x3, y3, test_size = 0.2, random_state = 0)

# Logistic Regression
# Test accuracy using split data
LR = LogisticRegression(random_state=0, solver='newton-cg', max_iter=300, multi_class='multinomial').fit(X_train3, y_train3)
LR.predict(X_test3)
print("Logistic Regression accuracy test: "+str(round(LR.score(X_test3,y_test3), 4)))

# Make prediction using test dataset given
LR = LogisticRegression(random_state=0, solver='newton-cg', max_iter=500, multi_class='multinomial').fit(x_real3, y_real3)
result = LR.predict(TestD3)
print("Logistic Regression prediction: ")
print(*result, sep = ", ")

# Support Vector Machine
# Test accuracy using split data
SVM = svm.SVC(kernel='linear', decision_function_shape="ovo").fit(X_train3, y_train3)
SVM.predict(X_test3)
print("Support Vector Machine accuracy test: "+str(round(SVM.score(X_test3, y_test3), 4)))

# Make prediction using test dataset given
SVM = svm.SVC(kernel='linear', decision_function_shape="ovo").fit(x_real3, y_real3)
result = SVM.predict(TestD3)
print("Support Vector Machine prediction: "+str(result))

# Random Forest
# Test accuracy using split data
RF = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(X_train3, y_train3)
RF.predict(X_test3)
print("Random Forest accuracy test: "+str(round(RF.score(X_test3, y_test3), 4)))

# Make prediction using test dataset given
RF = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(x_real3, y_real3)
result = RF.predict(TestD3)
print("Random Forest prediction"+str(result))

# Neural Network
# Test accuracy using split data
NN = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(150, 10), random_state=1).fit(X_train3, y_train3)
NN.predict(X_test3)
print("Neural Network accuracy test: "+str(round(NN.score(X_test3, y_test3), 4)))

# Make prediction using test dataset given
NN = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(150, 10), random_state=1).fit(x_real3, y_real3)
result = NN.predict(TestD3)
print("Neural Network prediction: "+str(result))