import pandas as pd
import numpy as np
import sklearn as sk

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

TD2 = pd.read_csv('Datasets/TrainData2.txt', sep="\t", header=None)
TL2 = pd.read_csv('Datasets/TrainLabel2.txt', sep="\t", header=None)
TestD2 = pd.read_csv('Datasets/TestData2.txt', sep="\t", header=None)

TD2.interpolate(axis=0, limit_direction="both")
TestD2.interpolate(axis=0, limit_direction="both")
TD2cat = pd.concat([TD2, TL2], axis=1)
print(TD2.shape)
print(TL2.shape)
print(TestD2.shape)

x2 = TD2cat.iloc[:,0:]
y2 = TD2cat.iloc[:,9182]

x_real2 = TD2.iloc[:,0:]
y_real2 = TL2.iloc[:,0]

X_train2, X_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size = 0.2, random_state = 0)

# Logistic Regression
# Test accuracy using split data
LR = LogisticRegression(random_state=0, solver='lbfgs', max_iter=300, multi_class='multinomial').fit(X_train2, y_train2)
LR.predict(X_test2)
print("Logistic Regression accuracy test: "+str(round(LR.score(X_test2,y_test2), 4)))

# Make prediction using test dataset given
LR = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, multi_class='multinomial').fit(x_real2, y_real2)
result = LR.predict(TestD2)
print("Logistic Regression prediction: "+str(result))

# Support Vector Machine
# Test accuracy using split data
SVM = svm.SVC(kernel='linear', decision_function_shape="ovo").fit(X_train2, y_train2)
SVM.predict(X_test2)
print("Support Vector Machine accuracy test: "+str(round(SVM.score(X_test2, y_test2), 4)))

# Make prediction using test dataset given
SVM = svm.SVC(kernel='linear', decision_function_shape="ovo").fit(x_real2, y_real2)
result = SVM.predict(TestD2)
print("Support Vector Machine prediction: "+str(result))

# Random Forest
# Test accuracy using split data
RF = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(X_train2, y_train2)
RF.predict(X_test2)
print("Random Forest accuracy test: "+str(round(RF.score(X_test2, y_test2), 4)))

# Make prediction using test dataset given
RF = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(x_real2, y_real2)
result = RF.predict(TestD2)
print("Random Forest prediction: "+str(result))

# Neural Network
# Test accuracy using split data
NN = MLPClassifier(solver='sgd', alpha=1e-5, activation='tanh', learning_rate='adaptive', hidden_layer_sizes=(150, 10), max_iter=1000, random_state=1).fit(X_train2, y_train2)
NN.predict(X_test2)
print("Neural Network accuracy test: "+str(round(NN.score(X_test2, y_test2), 4)))

# Make prediction using test dataset given
NN = MLPClassifier(solver='sgd', alpha=1e-5, activation='tanh', learning_rate='adaptive', hidden_layer_sizes=(150, 10),max_iter=1000, random_state=1).fit(x_real2, y_real2)
result = NN.predict(TestD2)
print("Neural Network prediction: "+str(result))