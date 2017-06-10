# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 23:42:51 2017

@author: user
"""

import pandas as pd
train1 = pd.read_csv("D:\Dummy.csv")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.ensemble import RandomForestClassifier


train, test = train_test_split(train1, test_size = 0.2)

feature1 = train.ix[:,6:20]
target1 = train["ShearTypeClass"].values

feature2 = test.ix[:,6:20]
target2= test["ShearTypeClass"].values

my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(feature1, target1)

my_prediction = my_tree_one.predict(feature2)
print(my_prediction)

print accuracy_score(my_prediction, target2)
#np.savetxt('vik2.csv',target2)
#np.savetxt('vik3.csv',my_prediction)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cnf_matrix = confusion_matrix(target2,my_prediction)
np.set_printoptions(precision=2)
print(cnf_matrix)
precision,recall,fscore,support=score(target2,my_prediction)
print('precision:{}'.format(precision))
print('recall:{}'.format(recall))
print('fscore:{}'.format(fscore))
print('support:{}'.format(support))



# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(feature1, target1)
pred_forest = my_forest.predict(feature2)
# Print the score of the fitted random forest
print accuracy_score(pred_forest, target2)
print(my_forest.score(feature1, target1))
# Compute confusion matrix
cnf_matrix = confusion_matrix(target2,pred_forest)
np.set_printoptions(precision=2)
print(cnf_matrix)
precision,recall,fscore,support=score(target2,pred_forest)
print('precision:{}'.format(precision))
print('recall:{}'.format(recall))
print('fscore:{}'.format(fscore))
print('support:{}'.format(support))
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(target2, pred_forest)
np.set_printoptions(precision=2)
class_names=[0,1,2,3]

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

#
#print(train.max())
#print(train.min())
#print(train.head())
#print(test.max())
#print(test.min())
#print(test.head())