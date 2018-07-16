import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.svm import SVC,LinearSVC,LinearSVR
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# 选取下面的8类
selected_categories = [
    'comp.graphics',
    'rec.motorcycles',
    'rec.sport.baseball',
    'misc.forsale',
    'sci.electronics',
    'sci.med',
    'talk.politics.guns',
    'talk.religion.misc']

# 加载数据集
newsgroups_train=fetch_20newsgroups(subset='train',
                                    categories=selected_categories,
                                    remove=('headers','footers','quotes'))
newsgroups_test=fetch_20newsgroups(subset='train',
                                    categories=selected_categories,
                                    remove=('headers','footers','quotes'))

train_texts=newsgroups_train['data']
train_labels=newsgroups_train['target']
test_texts=newsgroups_test['data']
test_labels=newsgroups_test['target']
print(len(train_texts),len(test_texts))

# 贝叶斯
text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=10000)),
                   ('clf',MultinomialNB())])
text_clf=text_clf.fit(train_texts,train_labels)
predicted=text_clf.predict(test_texts)
print("MultinomialNB准确率为：",np.mean(predicted==test_labels))

# SGD
text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=10000)),
                   ('clf',SGDClassifier())])
text_clf=text_clf.fit(train_texts,train_labels)
predicted=text_clf.predict(test_texts)
print("SGDClassifier准确率为：",np.mean(predicted==test_labels))

# LogisticRegression
text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=10000)),
                   ('clf',LogisticRegression())])
text_clf=text_clf.fit(train_texts,train_labels)
predicted=text_clf.predict(test_texts)
print("LogisticRegression准确率为：",np.mean(predicted==test_labels))

# SVM
text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=10000)),
                   ('clf',SVC())])
text_clf=text_clf.fit(train_texts,train_labels)
predicted=text_clf.predict(test_texts)
print("SVC准确率为：",np.mean(predicted==test_labels))

text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=10000)),
                   ('clf',LinearSVC())])
text_clf=text_clf.fit(train_texts,train_labels)
predicted=text_clf.predict(test_texts)
print("LinearSVC准确率为：",np.mean(predicted==test_labels))

text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=10000)),
                   ('clf',LinearSVR())])
text_clf=text_clf.fit(train_texts,train_labels)
predicted=text_clf.predict(test_texts)
print("LinearSVR准确率为：",np.mean(predicted==test_labels))

# MLPClassifier
text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=10000)),
                   ('clf',MLPClassifier())])
text_clf=text_clf.fit(train_texts,train_labels)
predicted=text_clf.predict(test_texts)
print("MLPClassifier准确率为：",np.mean(predicted==test_labels))

# KNeighborsClassifier
text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=10000)),
                   ('clf',KNeighborsClassifier())])
text_clf=text_clf.fit(train_texts,train_labels)
predicted=text_clf.predict(test_texts)
print("KNeighborsClassifier准确率为：",np.mean(predicted==test_labels))

# RandomForestClassifier
text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=10000)),
                   ('clf',RandomForestClassifier(n_estimators=8))])
text_clf=text_clf.fit(train_texts,train_labels)
predicted=text_clf.predict(test_texts)
print("RandomForestClassifier准确率为：",np.mean(predicted==test_labels))

# GradientBoostingClassifier
text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=10000)),
                   ('clf',GradientBoostingClassifier())])
text_clf=text_clf.fit(train_texts,train_labels)
predicted=text_clf.predict(test_texts)
print("GradientBoostingClassifier准确率为：",np.mean(predicted==test_labels))

# AdaBoostClassifier
text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=10000)),
                   ('clf',AdaBoostClassifier())])
text_clf=text_clf.fit(train_texts,train_labels)
predicted=text_clf.predict(test_texts)
print("AdaBoostClassifier准确率为：",np.mean(predicted==test_labels))

# DecisionTreeClassifier
text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=10000)),
                   ('clf',DecisionTreeClassifier())])
text_clf=text_clf.fit(train_texts,train_labels)
predicted=text_clf.predict(test_texts)
print("DecisionTreeClassifier准确率为：",np.mean(predicted==test_labels))


