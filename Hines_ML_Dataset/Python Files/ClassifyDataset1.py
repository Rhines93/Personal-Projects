import pandas as pd
import numpy as np
import sklearn as sk

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

TD1 = pd.read_csv('Datasets/TrainData1.txt', sep="\t", header=None)
TL1 = pd.read_csv('Datasets/TrainLabel1.txt', sep="\t", header=None)
TestD1 = pd.read_csv('Datasets/TestData1.txt', sep="\t", header=None)

TD1.fillna(TD1.median(), inplace=True)
TestD1.fillna(TestD1.median(), inplace=True)
TD1cat = pd.concat([TD1, TL1], axis=1)
print(TD1.shape)
print(TL1.shape)
print(TestD1.shape)

TD1.head()

x = TD1cat.iloc[:,0:]
y = TD1cat.iloc[:,3312]

x_real = TD1.iloc[:,0:]
y_real = TL1.iloc[:,0]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Logistic Regression
# Test accuracy using split data
LR = LogisticRegression(random_state=0, solver='newton-cg', max_iter=300, multi_class='multinomial').fit(X_train, y_train)
LR.predict(X_test)
print("Logistic Regression accuracy test: "+str(round(LR.score(X_test,y_test), 4)))

# Make prediction using test dataset given
LR = LogisticRegression(random_state=0, solver='lbfgs', max_iter=500, multi_class='multinomial').fit(x_real, y_real)
result = LR.predict(TestD1)
print("Logistic Regression: "+str(result))

# Support Vector Machine
# Test accuracy using split data
SVM = svm.SVC(kernel='linear', gamma='scale', decision_function_shape="ovo").fit(X_train, y_train)
SVM.predict(X_test)
print("Support Vector Machine accuracy test: "+str(round(SVM.score(X_test, y_test), 4)))

# Make prediction using test dataset given
SVM = svm.SVC(decision_function_shape="ovo").fit(x_real, y_real)
result = SVM.predict(TestD1)
print("Support Vector Machine prediction: "+str(result))

# Random Forest
# Test accuracy using split data
RF = RandomForestClassifier(n_estimators=1000, max_depth=15, random_state=0).fit(X_train, y_train)
RF.predict(X_test)
print("Random Forest accuracy test: "+str(round(RF.score(X_test, y_test), 4)))

# Make prediction using test dataset given
RF = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(x_real, y_real)
result = RF.predict(TestD1)
print("Random Forest prediction: "+str(result))

# Neural Network
# Test accuracy using split data
NN = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(150, 15), max_iter=500, random_state=1).fit(X_train, y_train)
NN.predict(X_test)
print("Neural Network accuracy test: "+str(round(NN.score(X_test, y_test), 4)))

# Make prediction using test dataset given
NN = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(150, 15), max_iter=500, random_state=1).fit(x_real, y_real)
result = NN.predict(TestD1)
print("Neural Network prediction: "+str(result))