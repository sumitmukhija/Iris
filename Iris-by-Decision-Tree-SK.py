import pandas as pd
import numpy as np
import csv
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

def create_test_and_train_set(features, label, test_size = 0.33):
	features_train, features_test, label_train, label_test = train_test_split(features, label, test_size = test_size, random_state=0, stratify = label)
	execute_classifier(label_test,label_train, features_test, features_train, test_size)

def execute_classifier(label_test, label_train, features_test, features_train, test_size):
	clf = tree.DecisionTreeClassifier()
	clf.fit(features_train, label_train)
	test_prediction = clf.predict(features_test)
	compute_accuracy(label_test, test_prediction, test_size)

def compute_accuracy(label_test, test_prediction, test_size):
	accuracy = accuracy_score(label_test, test_prediction)
	print("Accuracy: "+repr(round(accuracy * 100, 2))+r"% Testing data: "+repr(round(test_size * 100,2))+r"% ")
	with open('analysis.csv', 'a') as f:
		f.write(repr(round(test_size * 100,2))+","+repr(round(accuracy * 100, 2))+"\n")
	
#Prediction:
create_test_and_train_set(segregate_data(load_data())[0], segregate_data(load_data())[1])

#Checking for accuracy based on train-test split
# for i in range(95):
	# create_test_and_train_set(segregate_data(load_data())[0], segregate_data(load_data())[1], (i+3)/100)

