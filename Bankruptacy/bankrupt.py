import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")
bankrupt = pd.read_csv("file:///D:/DATA SCIENCE/ExcelR/Live Projects/Bankrupt prevention/bankruptcy-prevention.csv", sep = ';', header = 0)
bankrupt

bankrupt.describe()

bankrupt.shape

bankrupt.isnull().sum()

bankrupt_new = bankrupt.iloc[:,:]

bankrupt_new["class_yn"] = 1

bankrupt_new.loc[bankrupt[' class'] == 'bankruptcy', 'class_yn'] = 0

bankrupt_new.drop(' class', inplace = True, axis =1)
bankrupt_new.head()

# EDA

bankrupt_new.corr()
sns.heatmap(bankrupt_new.corr(), vmin = -1, vmax = 1, annot = True)

sns.countplot(x = 'class_yn', data = bankrupt_new, palette = 'hls')

sns.countplot(x = ' financial_flexibility', data = bankrupt_new, palette = 'hls')

# for visualization 

pd.crosstab(bankrupt.class_yn, bankrupt.industrial_risk).plot(kind='bar')

pd.crosstab(bankrupt_new[' financial_flexibility'], bankrupt_new['class_yn']).plot(kind = 'bar')

pd.crosstab(bankrupt_new[' credibility'], bankrupt_new.class_yn).plot(kind = 'bar')

pd.crosstab(bankrupt_new[' operating_risk'], bankrupt_new.class_yn).plot(kind='bar')

pd.crosstab(bankrupt_new[' financial_flexibility'], bankrupt_new[' credibility']).plot(kind = 'bar')

# model preparation

np.shape(bankrupt_new)

# Input
x = bankrupt_new.iloc[:,:-1]

# Target variable

y = bankrupt_new.iloc[:,-1]

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split # trian and test
from sklearn import metrics
from sklearn import preprocessing 
from sklearn.metrics import classification_report

# split the data into train and test

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 0)

logisticlassifier = LogisticRegression() 
logisticlassifier.fit(x_train, y_train)

logisticlassifier.coef_ # coefficients of features

#logisticlassifier.predict_proba (x_train) # probability values
# After the traing the model then we prediction on test data

y_pred = logisticlassifier.predict(x_test)

# let's test the performance of our model - confusion matrix

from sklearn.metrics import confusion_matrix

confusion_logist = confusion_matrix(y_test, y_pred)

confusion_logist

# accuracy of a model

# Train Accuracy

train_acc_logist = np.mean(logisticlassifier.predict(x_train)== y_train)

# Test Accuracy

test_acc_logist = np.mean(logisticlassifier.predict(x_test)== y_test)

from sklearn.metrics import accuracy_score

logistic_acc = accuracy_score(y_test, y_pred)
logistic_acc

##############

classifier = LogisticRegression()
classifier.fit(x, y)

classifier.coef_ # coefficients of features 
classifier.predict_proba (x) # Probability values 

y_pred = classifier.predict(x)

confusion_matrix = confusion_matrix(y, y_pred)
confusion_matrix

acc = accuracy_score(y, y_pred)
acc


## 2. KNN model

from sklearn.neighbors import KNeighborsClassifier 
import math
math.sqrt(len(y_test))

# Define the model KNN

KNN_classifier = KNeighborsClassifier(n_neighbors = 7, p =2, metric = 'euclidean')

# fit model

KNN_classifier.fit(x_train, y_train)

# Predict the test set results
test_acc = accuracy_score(KNN_classifier.predict(x_test),y_test)
train_acc = accuracy_score(neigh.predict(x_train), y_train)

y_pred = KNN_classifier.predict(x_test)
y_pred

# Evaluate model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import f1_score
print(f1_score(y_test, y_pred))

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test , y_pred))


# creating empty list variable
from sklearn.neighbors import KNeighborsClassifier as KNC

acc = []

# running KNN algorithm for 3 to 50 nearest neighbors (odd numbers) and 
# sorthing the accuracy values

for i in range(3,50,2):
    neigh = KNC(n_neighbors = i)
    neigh.fit(x_train, y_train)
    train_acc = accuracy_score(neigh.predict(x_train), y_train)
    test_acc = accuracy_score(neigh.predict(x_test), y_test)
    acc.append([train_acc, test_acc])    
# train accuracy plot
plt.plot(np.arange(3,50,2),[i[0] for i in acc], "ro-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc], "bo-")

plt.legend(["train", "test"])    


##################################3
#############

# 3. Naive Bayes Classifier

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB

# Creating GaussianNB and MultinomialNB functions

GNB = GaussianNB()
MNB = MultinomialNB()

# Building the model with GaussianNB

Naive_GNB = GNB.fit(x_train ,y_train)

y_pred = Naive_GNB.predict(x_test)
y_pred

# Evaluate model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test , y_pred))

# Building the model with GaussianNB

Naive_MNB = MNB.fit(x_train ,y_train)

y_pred = Naive_MNB.predict(x_test)
y_pred

# Evaluate model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test , y_pred))

################ Support Vector Machine ##########3

from sklearn import svm

# importing scikit learn with make_blobs
from sklearn.datasets.samples_generator import make_blobs

# creating datasets p containing n_samples
# q containing two classes
p, q = make_blobs(n_samples=300, centers=2,
				random_state=0, cluster_std=0.40)
import matplotlib.pyplot as plt
# plotting scatters
plt.scatter(p[:, 0], p[:, 1], c=q, s=50, cmap='spring');
plt.show()

# import support vector classifier
# "Support Vector Classifier"
from sklearn.svm import SVC
SVM_classifier = SVC(kernel='linear')

# fitting x samples and y classes
SVM_classifier.fit(x, y)

SVM_classifier.predict([[0,0.5,1,0.5,0.5,0.5]])

SVM_classifier.predict([[1,1,0,0.5,0,1]])

y_pred = SVM_classifier.predict(x)

from sklearn.metrics import accuracy_score
svm_acc = accuracy_score(y, y_pred)
svm_acc

# we got the accuracy as 99.6 % overfitted to avoid overfitting we use regularization

from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha = 50, max_iter = 100, tol = 0.1)

ridge_reg.fit(x_train, y_train)

ridge_reg.score(x_test, y_test)

ridge_reg.score(x_train, y_train)








######## 
# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(x_train,y_train)
pred_test_linear = model_linear.predict(x_test)

np.mean(pred_test_linear==y_test)

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(x_train,y_train)
pred_test_poly = model_poly.predict(x_test)

np.mean(pred_test_poly==y_test) # Accuracy = 95.238

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(x_train,y_train)
pred_test_rbf = model_rbf.predict(x_test)

np.mean(pred_test_rbf==y_test) # Accuracy = 100%

# from this we choose the kernal poly is the best method


################ Decision Tree ###############

from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree

model_tree = DecisionTreeClassifier(criterion = 'entropy')

model_tree = model_tree.fit(x_train, y_train)

y_pred = model_tree.predict(x_test)

# Accuracy = train
np.mean(model_tree.predict(x_train)== y_train)

# Accuracy = Test
np.mean(y_pred==y_test) # 1

tree.plot_tree(model_tree.fit(x_train, y_train))

import graphviz 
dot_data = tree.export_graphviz(model_tree, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("bankrupt_new") 

dot_data = tree.export_graphviz(model_tree, out_file=None, 
                      feature_names=x,  
                      class_names=y,  
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 

import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(15,10))
tree.plot_tree(model_tree, filled = True)
plot_decision_tree(model_tree, x_train.columns, bankrupt_new.columns[6])

############# Random Forest Classifier ##############

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=10,criterion="entropy")
# n_estimators -> Number of trees ( you can increase for better accuracy)
# n_jobs -> Parallelization of the computing and signifies the number of jobs 
# running parallel for both fit and predict
# oob_score = True means model has done out of box sampling to make predictions

rf.fit(x_train, y_train) # Fitting RandomForestClassifier model from sklearn.ensemble 
rf.estimators_ # 
rf.classes_ # class labels (output)
rf.n_classes_ # Number of levels in class labels 
rf.n_features_  # Number of input features in model 8 here.

rf.n_outputs_ # Number of outputs when fit performed

rf.oob_score_  # 0.72916

rf_train_pred = rf.predict(x_train)
rf_test_pred = rf.predict(x_test)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(rf_test_pred, y_test))

train_acc = accuracy_score(rf_train_pred, y_train)
train_acc

test_acc = accuracy_score(rf_test_pred, y_test)
test_acc

##########

rf.fit(x, y)

y_pred = rf.predict(x)
print(confusion_matrix(y_pred, y))
print(classification_report(y_pred, y))
print(accuracy_score(y_pred, y))

# Visualising the Random Forest Regression results

# arange for creating a range of values
# from min value of x to max
# value of x with a difference of 0.01
# between two consecutive values
X_grid = np.arange(min(x), max(x), 0.01)

# reshape for reshaping the data into a len(X_grid)*1 array,
# i.e. to make a column out of the X_grid value				
X_grid = X_grid.reshape((len(X_grid), 1))

# Scatter plot for original data
plt.scatter(x, y, color = 'blue')

# plot predicted data
plt.plot(X_grid, regressor.predict(X_grid),
		color = 'green')
plt.title('Random Forest Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

########### Neural Network ##########3

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(15,15))

mlp.fit(x_train,y_train)
prediction_train=mlp.predict(x_train)
prediction_test = mlp.predict(x_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,prediction_test))
np.mean(y_test==prediction_test)
np.mean(y_train==prediction_train)

############ AutoML techniques ###############3

## Autosklearn  model

import autosklearn
print('autosklearn: %s' % autosklearn.__version__)



