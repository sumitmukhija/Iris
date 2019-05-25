import pandas as pd
import numpy as np
from sklearn import preprocessing, tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def load_data():
	df = pd.read_csv('Iris.data')
	return df

def segregate_data(df):
	features = df.iloc[:,[0,1,2,3]]
	textual_label = df.iloc[:,[4]]
	le = preprocessing.LabelEncoder()
	#0: Iris-setosa 1: Iris-versicolor 2: Iris-virginica
	label = le.fit_transform(np.ravel(textual_label))
	return (features, label)

def create_test_and_train_set(features, label):
	features_train, features_test, label_train, label_test = train_test_split(features, label, test_size = 0.33, random_state=0, stratify = label)
	execute_classifier(label_test,label_train, features_test, features_train)

def execute_classifier(label_test, label_train, features_test, features_train):
	clf = tree.DecisionTreeClassifier()
	clf.fit(features_train, label_train)
	test_prediction = clf.predict(features_test)
	compute_accuracy(label_test, test_prediction)

def compute_accuracy(label_test, test_prediction):
	accuracy = accuracy_score(label_test, test_prediction)
	print("Accuracy is "+repr(accuracy * 100)+"% ")

create_test_and_train_set(segregate_data(load_data())[0], segregate_data(load_data())[1])

