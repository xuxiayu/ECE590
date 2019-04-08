import csv
from copy import deepcopy
import operator
import time
start_time = time.time()
import numpy as np

 
increment_cost = [19296, 19296, 110592, 110592, 491520, 491520, 491520, \
					 1966080, 1966080, 1966080, 2359296, 2359296, 2359296, 5120]
total_bits = 1.3*sum(increment_cost)
max_precision = 4
accuracy_mat = [[0] for x in range(14)]
with open('quantized_data.csv') as csvDataFile:
	csvReader = csv.reader(csvDataFile)
	ind = 0
	for row in csvReader:
		accuracy_mat[ind]=list(map(float,row[0:]))
		ind += 1

shape = [max_precision]*14
visited_mat = np.zeros(shape=shape, dtype=bool)
final_mat = np.zeros(shape=shape, dtype=bool)
working_mat = np.zeros(shape=shape, dtype=bool)

def find_bits(scheme):
	used_bits = sum([a*b for a,b in zip(scheme,increment_cost)])
	return used_bits

def find_accuracy(test_weights):
	accuracy = [0]*14
	for i in range(14):
		if test_weights[i]>1:
			accuracy[i]=accuracy_mat[i][test_weights[i]-1]-accuracy_mat[i][0]
		else:
			accuracy[i] = 0
	return sum(accuracy)

def mat_get(mat,weights):
	x = [weight-1 for weight in weights]
	return mat[x[0]][x[1]][x[2]][x[3]][x[4]][x[5]][x[6]][x[7]][x[8]][x[9]][x[10]][x[11]][x[12]][x[13]]

def mat_set(mat,weights):
	x = [weight-1 for weight in weights]
	mat[x[0]][x[1]][x[2]][x[3]][x[4]][x[5]][x[6]][x[7]][x[8]][x[9]][x[10]][x[11]][x[12]][x[13]] = 1

# Initialize list of quantization schemes        
def find_scheme():
	best_weights = [1]*14
	if sum(increment_cost)+min(increment_cost)<=total_bits:
		working_list = []
		best_accuracy = 0
		final_list = []
		working_list.append(best_weights)
		while len(working_list)>0:
			print(best_weights,best_accuracy,len(working_list))
			template_scheme = working_list.pop()
			final_scheme = True
			mat_set(visited_mat,template_scheme)
			for layer in range(14):
				current_scheme = template_scheme.copy()
				current_scheme[layer] += 1
				if (increment_cost[layer] + find_bits(template_scheme) <= total_bits) \
					and (template_scheme[layer] < max_precision)\
					and mat_get(working_mat,current_scheme) != 1 and \
					   	mat_get(visited_mat,current_scheme) != 1:
					final_scheme = False
					working_list.append(current_scheme)
					mat_set(working_mat,template_scheme)
			# Couldn't increment scheme further, pushed final version to list
			if final_scheme == True:
				template_accuracy = find_accuracy(template_scheme)
				if template_accuracy>best_accuracy:
					best_weights = template_scheme
					best_accuracy = template_accuracy
		return best_weights
	elif sum(increment_cost)<=total_bits:
		return best_weights
	else:
		raise Exception('No quantization scheme could be found for the specified number of total bits.')
	print(best_weights)
	print("--- %s seconds ---" % (time.time() - start_time))