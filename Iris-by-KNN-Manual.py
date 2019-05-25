# Using KNN for Iris data set.
import csv
import random
import math
import operator

def load_data(split):
	test_set, training_set = [],[]
	with open('iris.data') as data_file:
		lines = csv.reader(data_file)
		data = list(lines)
		threshold = int((split/100) * len(data))
		
		for i in range(len(data) - 1):
			for j in range(4):
				data[i][j] = float(data[i][j])
			if i < threshold:
				training_set.append(data[i])
			else:
				test_set.append(data[i])
		return (training_set, test_set)


def get_euclidean_distance(instance_one, instance_two, length):
	NUMBER_OF_FEATURES_TO_BE_CONSIDERED = 3
	computed_distance = 0
	for i in range(length):
		computed_distance = computed_distance + pow(instance_one[i] - instance_two[i], 2)
		computed_distance = math.sqrt(computed_distance)
	return computed_distance


def get_k_neighbors(training_set, test_instance, k):
    distances = []
    length = len(test_instance) - 1
    for i in range(len(training_set)):
        dist = get_euclidean_distance(test_instance, training_set[i], length)
        distances.append((training_set[i], dist)) #Adding tupple
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def get_accuracy(test_set):
	correct = 0
	predictions = []
	for i in range(len(test_set)):
		neighbours = get_k_neighbors(training_set, test_set[i], 3)
		result = getResponse(neighbours)
		if test_set[i][-1] == result:
			correct += 1
	return (correct/float(len(test_set))) * 100


lines = []
lines.append(['Accuracy', 'Training values', 'Test values', 'Training percentage'])
for i in range(98):
	split = i + 2
	training_set, test_set = load_data(split)
	neighbours = get_k_neighbors(training_set, [1,2,3,0], 3)
	print("Result: "+ getResponse(neighbours))
	accuracy = get_accuracy(test_set)
	row = []
	row.append(round(accuracy,2))
	row.append(len(training_set)) 
	row.append(len(test_set)) 
	row.append(split)
	lines.append(row)
	print("Accuracy: "+repr(accuracy)+"% Training: "+repr(len(training_set))+" Test: "+repr(len(test_set))+" Split: "+repr(split))
with open('split_dist.csv','w') as wF:
	writer = csv.writer(wF)
	writer.writerows(lines)



