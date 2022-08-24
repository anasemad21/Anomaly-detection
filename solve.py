import pandas as pd
from sklearn import  tree
import numpy as np
from scipy.stats import chisquare
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector
file=pd.read_csv("data.csv")
features_x = file.iloc[:, 2:]
features_y = file.iloc[:, 1]
features_y_replaced = features_y.replace(("M", "B"), (1, 0))
def fillter_method(file):
    f=open("report.txt","w")
    f.write("")
    f.close()
    accepted_features=[]#list of colums names that have p value more than 0.05 and this will be accepted coloum to us
    for i in range(features_x.shape[1]):
         chi,pvalue=chisquare(features_x.iloc[:,i:i+1])
         if pvalue >0.05:
             accepted_features.extend(features_x.iloc[:, i:i + 1])
    f = open("report.txt", "w")
    f.write("filter_method")
    f.write("\n")
    f.write(f"{accepted_features}")
    f.write("\n")
    f.write(f"{len(accepted_features)}")
    f.write("\n")
    f.close()
    #print(accepted_features)
    columns=list(file.columns.values.tolist())# return list of coloums names in file
    for i in columns:# deleting colums that we are not interested on it
        if i not in accepted_features:
            del (file[i])
    return file

def wrapperSelection(x,y):
    f = open("report.txt", "w")
    f.write("")
    f.close()
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x,y)
    bs=SequentialFeatureSelector(knn,k_features='best',forward=False,n_jobs=-1)
    bs.fit(x,y)
    feature=list(bs.k_feature_names_)
    f = open("report.txt", "w")
    f.write("wrapper_method")
    f.write("\n")
    f.write(f"{feature}")
    f.write("\n")
    f.write(f"{len(feature)}")
    f.write("\n")
    f.write(f"{knn}")
    f.write("\n")
    f.close()
    return feature


#knn = KNeighborsClassifier(n_neighbors=3)


def select_colums_wrapper(file,p):
    columns = list(file.columns.values.tolist())  # return list of coloums names in file
    for i in columns:  # deleting colums that we are not interested on it
        if i not in p:
            del (file[i])
    return file

p=wrapperSelection(features_x,features_y_replaced)


fille=select_colums_wrapper(file,p)

#fille=fillter_method(file)

x_train, x_test, y_train, y_test = train_test_split(fille, features_y_replaced, test_size=0.4)
# spilting data 60% train 40%test


def KNN(x_train, x_test, y_train, y_test):
    n = KNeighborsClassifier(n_neighbors=3)
    n.fit(x_train, y_train)
    y_predict=n.predict(x_test)
    pre=precision_score(y_test,y_predict)
    rec=recall_score(y_test,y_predict)
    accuracy = accuracy_score(y_test,y_predict)
    measure = 2 * (pre * rec) / (pre + rec)
    return pre, rec, accuracy, measure

def NBayes(x_train, x_test, y_train, y_test ):
    gnb = GaussianNB()
    gnb.fit(x_train,y_train.values)
    y_predict = gnb.predict(x_test)
    pre = precision_score(y_test ,y_predict)
    rec = recall_score(y_test ,y_predict)
    score = accuracy_score(y_test, y_predict)
    measure=2*(pre*rec)/(pre+rec)
    return pre, rec,score,measure
def logistic (X_train, X_test, y_train, y_test ):
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train,y_train)
    y_predict = classifier.predict(X_test)
    pre = precision_score(y_test,y_predict)
    rec = recall_score(y_test, y_predict)
    score = accuracy_score(y_test, y_predict)
    measure = 2 * (pre * rec) / (pre + rec)
    return pre, rec, score, measure

print("logistic: ",logistic(x_train, x_test, y_train, y_test))
print("NBayes: ",NBayes(x_train, x_test, y_train, y_test))
print("KNN:  ",KNN(x_train, x_test, y_train, y_test))
