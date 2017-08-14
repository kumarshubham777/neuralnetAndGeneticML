from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import metrics
import ga as ga
__author__ = 'Shubham'
from sklearn.model_selection import cross_val_score,KFold
import NN as NN
cnf=np.zeros(2)
accuracy=0

#print(kf.split(NN.x))
kf=cross_validation.KFold(250,n_folds=10,shuffle=False,random_state=None)
for train,test in kf:


    y_test=NN.y.loc[test]
    y_pred=ga.genetic(NN.x,NN.y,NN.x.loc[test])
    confusion = confusion_matrix(y_test,y_pred)
    cnf=cnf+confusion
    accuracy =accuracy+ accuracy_score(y_test, y_pred)


#print(confusion)
print ("Confusion matrix : ",cnf)
print("Accuracy is :::: ",accuracy/10)
print("Error is :::: ",1- accuracy/10)



