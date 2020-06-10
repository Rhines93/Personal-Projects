import pandas as pd
import numpy as np
import sklearn as sk

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

TD4 = pd.read_csv('Datasets/TrainData4.txt', sep="\t", header=None)
TL4 = pd.read_csv('Datasets/TrainLabel4.txt', sep="\t", header=None)
TestD4 = pd.read_csv('Datasets/TestData4.txt', sep="\t", header=None)

TD4.fillna(TD4.median(), inplace=True)
TestD4.fillna(TestD4.median(), inplace=True)
TD4cat = pd.concat([TD4, TL4], axis=1)
print(TD4.shape)
print(TL4.shape)
print(TestD4.shape)

x4 = TD4cat.iloc[:,0:]
y4 = TD4cat.iloc[:,112]

x_real4 = TD4.iloc[:,0:]
y_real4 = TL4.iloc[:,0]

X_train4, X_test4, y_train4, y_test4 = train_test_split(x4, y4, test_size = 0.2, random_state = 0)

# Logistic Regression
# Test accuracy using split data
LR = LogisticRegression(random_state=0, solver='newton-cg', class_weight='balanced', max_iter=500, multi_class='multinomial').fit(X_train4, y_train4)
LR.predict(X_test4)
print("Logistic Regression accuracy test: "+str(round(LR.score(X_test4,y_test4), 4)))

# Make prediction using test dataset given
LR = LogisticRegression(random_state=0, solver='newton-cg', max_iter=500, multi_class='multinomial').fit(x_real4, y_real4)
result = LR.predict(TestD4)
print("Logistic Regression prediction: "+str(result))

# Support Vector Machine
# Test accuracy using split data
SVM = svm.SVC(kernel='rbf', decision_function_shape="ovo").fit(X_train4, y_train4)
SVM.predict(X_test4)
print("Support Vector Machine accuracy test: "+str(round(SVM.score(X_test4, y_test4), 4)))

# Make prediction using test dataset given
SVM = svm.SVC(decision_function_shape="ovo").fit(x_real4, y_real4)
result = SVM.predict(TestD4)
print("Support Vector Machine prediction: "+str(result))

# Random Forest
# Test accuracy using split data
RF = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(X_train4, y_train4)
RF.predict(X_test4)
print("Random Forest accuracy test: "+str(round(RF.score(X_test4, y_test4), 4)))

# Make prediction using test dataset given
RF = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(x_real4, y_real4)
result = RF.predict(TestD4)
print("Random Forest prediction: "+str(result))

# Neural Network
# Test accuracy using split data
NN = MLPClassifier(solver='adam', alpha=50, max_iter=500, hidden_layer_sizes=(150, 10), random_state=1).fit(X_train4, y_train4)
NN.predict(X_test4)
print("Neural Network accuracy test: "+str(round(NN.score(X_test4, y_test4), 4)))

# Make prediction using test dataset given
NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 10), random_state=1).fit(x_real4, y_real4)
result = NN.predict(TestD4)
print("Neural Network prediction: "+str(result))