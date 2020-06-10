import pandas as pd
import numpy as np
import sklearn as sk

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

TD5 = pd.read_csv('TrainData5.txt', sep="\t", header=None)
TL5 = pd.read_csv('TrainLabel5.txt', sep="\t", header=None)
TestD5 = pd.read_csv('TestData5.txt', sep="\t", header=None)

TD5.fillna(TD5.median(), inplace=True)
TestD5.fillna(TestD5.median(), inplace=True)
TD5cat = pd.concat([TD5, TL5], axis=1)
print(TD5.shape)
print(TL5.shape)
print(TestD5.shape)

x5 = TD5cat.iloc[:,0:]
y5 = TD5cat.iloc[:,11]

x_real5 = TD5.iloc[:,0:]
y_real5 = TL5.iloc[:,0]

X_train5, X_test5, y_train5, y_test5 = train_test_split(x5, y5, test_size = 0.2, random_state = 0)

# Logistic Regression
# Test accuracy using split data
LR = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, multi_class='multinomial').fit(X_train5, y_train5)
LR.predict(X_test5)
print("Logistic Regression accuracy test: "+str(round(LR.score(X_test5,y_test5), 4)))

# Make prediction using test dataset given
LR = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, multi_class='multinomial').fit(x_real5, y_real5)
result = LR.predict(TestD5)
print("Logistic Regression prediction: "+str(result))

# Support Vector Machine
# Test accuracy using split data
SVM = svm.SVC(kernel='linear', decision_function_shape="ovo").fit(X_train5, y_train5)
SVM.predict(X_test5)
print("Support Vector Machine accuracy test: "+str(round(SVM.score(X_test5, y_test5), 4)))

# Make prediction using test dataset given
SVM = svm.SVC(kernel='linear', decision_function_shape="ovo").fit(x_real5, y_real5)
result = SVM.predict(TestD5)
print("Support Vector Machine prediction: "+str(result))

# Random Forest
# Test accuracy using split data
RF = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(X_train5, y_train5)
RF.predict(X_test5)
print("Random Forest accuracy test: "+str(round(RF.score(X_test5, y_test5), 4)))

# Make prediction using test dataset given
RF = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(x_real5, y_real5)
result = RF.predict(TestD5)
print("Random Forest prediction: "+str(result))

# Neural Network
# Test accuracy using split data
NN = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(150, 15), max_iter=500, random_state=1).fit(X_train5, y_train5)
NN.predict(X_test5)
print("Neural network accuracy test: "+str(round(NN.score(X_test5, y_test5), 4)))

# Make prediction using test dataset given
NN = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(150, 15), max_iter=500, random_state=1).fit(x_real5, y_real5)
result = NN.predict(TestD5)
print("Neural network prediction: "+str(result))