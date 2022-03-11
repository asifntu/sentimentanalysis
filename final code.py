

import itertools
import pandas as pd
import csv
import xlrd
import os,sys
import numpy as np
from numpy import std
non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)
from xlutils.copy import copy
from xlrd import *
from sklearn.metrics import confusion_matrix
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from numpy import nan
data = pd.read_csv("prec4.csv",encoding='latin-1')
data.replace(nan, 'Neutral', inplace=True)
X=data.TITLE
#data['NUM_CATEGORY']=data.CATEGORY.map({'Moderate':0,'Neutral':1,'low extreme':2,'highly extreme':3})
#y=data.NUM_CATEGORY
y=data.CATEGORY

# shuffle
'''
idx = np.arange(X.shape[0])
np.random.seed(13)
np.random.shuffle(idx)
X = X[idx]
y = y[idx]
'''
# standardize

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)


'''
X_train_counts.shape

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

'''
'''
X=np.array(X)
mean = np.mean(X,axis=0)
std = np.std(X,axis=0)
X = (X - mean) / std
h = .02  # step size in t

'''






print(data.CATEGORY.unique())
print(data.groupby('CATEGORY').describe())


print(data.CATEGORY.count())


#data['NUM_CATEGORY']=data.CATEGORY.map({'Moderate':0,'Neutral':1,'low extreme':2,'highly extreme':3})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(ngram_range=(1,1))
X_train_counts = count_vect.fit_transform(X_train)
print("training features")
#print len(X_train_counts)
print(X_train_counts.shape)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer(use_idf=False)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print("tf-idf  featturees  ")
print(X_train_tfidf.shape)
from sklearn.feature_extraction.text import HashingVectorizer
vectorizer = HashingVectorizer(n_features=2**4)
X_train_hashing = vectorizer.fit_transform(X_train)
print(X_train_hashing.shape)
from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile, chi2
#X_new = SelectKBest(chi2, k=5).fit_transform(X_train,y_train)
#ch2 = SelectKBest(chi2, k=4000)
ch2= SelectPercentile(chi2, percentile=100)
X_train_ch = ch2.fit_transform(X_train_tfidf , y_train)
#X_test = ch2.transform(X_test)
#print(X_new.shape)

'''
'''
print(y_train.describe())
print(y_test.describe())

from yellowbrick.classifier import ClassificationReport
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=0.01).fit(X_train_tfidf, y_train)


from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2),stop_words='english')),
                     ('tfidf', TfidfTransformer(use_idf=False)),
                     ('clf', MultinomialNB(alpha=0.01))
                     ])
text_clf = text_clf.fit(X_train, y_train)


import numpy as np

predicted = text_clf.predict(X_test)

print("multinomial nb")
print(np.mean(predicted == y_test))
from sklearn import metrics
print(metrics.classification_report(y_test, predicted,data.CATEGORY.unique()))
print(metrics.confusion_matrix(y_test, predicted))


'''

from sklearn.naive_bayes import BernoulliNB
#clf = MultinomialNB(alpha=0.01).fit(X_train_tfidf, y_train)


from sklearn.pipeline import Pipeline
text_clfber = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2),stop_words='english')),
                     ('tfidf', TfidfTransformer(use_idf=False)),
                     ('clfber', BernoulliNB(alpha=0.01, binarize=0.0, class_prior=None, fit_prior=True)),
                     ])
text_clfber = text_clfber.fit(X_train, y_train)


import numpy as np

predicted = text_clfber.predict(X_test)
print("bernolli nb")
print(np.mean(predicted == y_test))

from sklearn.naive_bayes import GaussianNB
#clfgau = GaussianNB(alpha=0.01).fit(X_train_tfidf, y_train)

from sklearn.pipeline import Pipeline
text_clfgau = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2),stop_words='english')),
                     ('tfidf', TfidfTransformer(use_idf=False)),
                     ('clfgau', GaussianNB(priors=None, var_smoothing=1e-09)),
                     ])
text_clfgau = text_clfgau.fit(X_train, y_train)


import numpy as np

predictedgau = text_clfgau.predict(X_test)
print("gaussian nb")
print(np.mean(predictedgau == y_test))


'''


'''
plt.plot(y_test, predicted)
plt.xlabel("True Values")
plt.ylabel("Predicted")
plt.show()
'''
'''

from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1))),
                     ('tfidf', TfidfTransformer(use_idf=False)),
                     ('clf-svm', SGDClassifier(loss='log', penalty='l1',
                                           alpha=1e-5, max_iter=50, random_state=42)),
 ])
text_clf_svm= text_clf_svm.fit(X_train, y_train)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


'''
'''
print(text_clf_svm)
predicted_svm = text_clf_svm.predict(X_test)
print("predicted_svm")
print(np.mean(predicted_svm == y_test))
#print(predicted_svm)
'''
'''

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
Z = text_clf_svm.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.axis('tight')

# Plot also the training points
for i, color in zip(text_clf_svm.classes_, colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, label=data.category_names[i],
                cmap=plt.cm.Paired, edgecolor='black', s=20)
plt.title("Decision surface of multi-class SGD")
plt.axis('tight')

# Plot the three one-against-all classifiers
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
coef = text_clf_svm.coef_
intercept = text_clf_svm.intercept_


def plot_hyperplane(c, color):
    def line(x0):
        return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]

    plt.plot([xmin, xmax], [line(xmin), line(xmax)],
             ls="--", color=color)


for i, color in zip(text_clf_svm.classes_, colors):
    plot_hyperplane(i, color)
plt.legend()
plt.show()





'''
'''
#print(predicted_svm.groupby('CATEGORY').describe())
print("count of y test")
print(y_test.value_counts())
print(confusion_matrix(y_test,predicted_svm))

print(np.mean(predicted_svm == y_test))
unique_label = np.unique(y_test)
print(unique_label)

cm=pd.DataFrame(confusion_matrix(y_test,predicted_svm, labels=unique_label), 
                   index=['true:{:}'.format(x) for x in unique_label], 
                   columns=['pred:{:}'.format(x) for x in unique_label])
print(cm)
#print(confusion_matrix(y_test,predicted_svm, labels=unique_label))
#plt.plot(cm)

plt.show()
cm.plot()
plt.show()


#This function return the class of the input news
def predict1_news(news):
    #test = count_vect.transform(news)

    #test= tfidf_transformer.transform(news)
    pred= text_clf_svm.predict(news)
    if pred  == 0:
         return 'Neutral'
    if pred == 1:
        return 'Moderate'
    if pred == 2:
        return 'low extreme'
    elif pred == 3:
        return 'highly extreme'
    
##Copy and paste the news headline in 'x'
#with open('chec.csv') as File:  
    #reader = csv.reader(File)
    ##for row in reader:
workbook = xlrd.open_workbook('chec2.xls')
worksheet = workbook.sheet_by_index(0)
w = copy(open_workbook('chec2.xls'))
num_rows=worksheet.nrows
# Value of 1st row and 1st column
rows=[]
for row in range(1,num_rows):
  values=[]
  value=(worksheet.cell(row, 2).value)
  x=[value]
  r = predict1_news(x)
  w.get_sheet(0).write(row,4,r)
  #print (value.translate(non_bmp_map) + "prediction " + r)
w.save('chec2.xls')
'''
'''
#Printing the confusion matrix of our prediction
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, result)
'''
'''

'''

'''
from sklearn import svm
text_clf_svc = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1))),
                     ('tfidf', TfidfTransformer(use_idf=False)),
                     ('clf-svc', svm.SVC(C=1.0, kernel='linear'
   )),
])
'''
'''
#clf-svc', svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False))
    '''
'''
text_clf_svc= text_clf_svc.fit(X_train, y_train)

print(text_clf_svc)
predicted_svc = text_clf_svc.predict(X_test)
print("svc svm")
print(np.mean(predicted_svc == y_test))

'''


from sklearn.svm import LinearSVC


text_clf_linsvc = Pipeline([('vect', CountVectorizer(ngram_range=(1,1))),
                      ('tfidf', TfidfTransformer(use_idf=False)),
                     #(#'ch2 ', SelectPercentile(chi2, percentile=100)),
                     ('clf_svc', LinearSVC(penalty="l1", dual=False,
                                                  tol=1e-3)), 
                     #('relif', ReliefF(n_features_to_select=2, n_neighbors=100))
                      
                            
])
from sklearn.model_selection import cross_val_score
scoresv = cross_val_score(text_clf_linsvc , X, y, cv=10)
print(scoresv)
from sklearn.feature_selection import RFE

text_clf_linsvc= text_clf_linsvc.fit(X_train, y_train)

print(text_clf_linsvc)
predicted_linsvc = text_clf_linsvc.predict(X_test)
print(predicted_linsvc)
print("linear svm")
print(np.mean(predicted_linsvc == y_test))
from sklearn import metrics
print(metrics.classification_report(y_test, predicted_linsvc,data.CATEGORY.unique()))
print(metrics.confusion_matrix(y_test, predicted_linsvc))



'''
def predict_news(news):
    #test = count_vect.transform(news)
    pred= text_clf_linsvc.predict(news)
    #print(news)
    print(pred)
workbook = xlrd.open_workbook('validate.xls')
worksheet = workbook.sheet_by_index(0)
w = copy(open_workbook('validate.xls'))
num_rows=worksheet.nrows
# Value of 1st row and 1st column
rows=[]
for row in range(0,num_rows):
    values=[]
    value=(worksheet.cell(row, 4).value)
    x=[value]
    r = predict_news(x)
    #w.get_sheet(0).write(row,2,r)
    #print (value)
    #print(r)
#w.save('validate.xls')
'''
'''

'''
'''
'''
'''
from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
               'tfidf__use_idf': (True, False),
               'clf__alpha': (1e-2, 1e-3),
}
gs_clf = GridSearchCV(text_clf, parameters)
gs_clf = gs_clf.fit(X_train, y_train)
print("Naive Bayes")
#print(gs_clf.best_score_)
#print(gs_clf.best_params_)
'''
'''
from sklearn.model_selection import GridSearchCV
parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)],
               'tfidf__use_idf': (True, False),
               'clf-svm__alpha': (1e-2, 1e-5),
 }
gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm)
gs_clf_svm = gs_clf_svm.fit(X_train, y_train)
print("SVM Classifier")

print(gs_clf_svm.best_score_)
print(gs_clf_svm.best_params_)

'''

