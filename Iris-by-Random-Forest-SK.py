import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
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
	execute_classifier(label_test, label_train, features_test, features_train, test_size)

def execute_classifier(label_test, label_train, features_test, features_train, test_size):
	clf = RandomForestClassifier(n_estimators = 10)
	clf.fit(features_train,  np.ravel(label_train))
	predict_sample(clf)
	print("Accuracy: "+repr(round(clf.score(features_test, label_test) * 100, 2)) + "% Test size: "+repr(round(test_size * 100, 2))+"%")

def predict_sample(rf):
	prediction = rf.predict([[1.0,3.2,31,0]])
	prediction_post_string = "setosa" if prediction == 0 else "versicolor" if prediction == 1 else "virginica"
	print("Iris-"+prediction_post_string)


features, label = segregate_data(load_data())
np.random.seed(0)
create_test_and_train_set(features, label)