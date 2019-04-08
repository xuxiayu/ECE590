import csv
from copy import deepcopy
import operator
import time
import itertools
start_time = time.time()

 
increment_cost = [19296, 19296, 110592, 110592, 491520, 491520, 491520, \
					 1966080, 1966080, 1966080, 2359296, 2359296, 2359296, 5120]

total_bits = 2*sum(increment_cost)
max_precision = 4
accuracy_mat = [[0] for x in range(14)]
with open('quantized_data.csv') as csvDataFile:
	csvReader = csv.reader(csvDataFile)
	ind = 0
	for row in csvReader:
		accuracy_mat[ind]=list(map(float,row[0:]))
		ind += 1

def find_scheme():
	best_weights = [1]*14
	if sum(increment_cost)+min(increment_cost)<=total_bits:
		accuracy = [0]*14
		counter = 0
		candidate_list = []
		best_weights = [1]*14
		best_accuracy = 0
		for l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14 in itertools.product(range(1,max_precision+1),repeat=14):
			test_weights = [l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14]
			used_bits = sum([a*b for a,b in zip(test_weights,increment_cost)])
			if (used_bits <= total_bits) and (used_bits/total_bits)>.75:
				for i in range(14):
					if test_weights[i]>1:
						accuracy[i]=accuracy_mat[i][test_weights[i]-1]-accuracy_mat[i][0]
					else:
						accuracy[i] = 0
				test_accuracy = sum(accuracy)
				if test_accuracy>best_accuracy:
					best_weights = test_weights
					best_accuracy = test_accuracy
	elif sum(increment_cost)<=total_bits:
		return best_weights
	else:
		raise Exception('No quantization scheme could be found for the specified number of total bits.')
	print(best_weights)
	print("--- %s seconds ---" % (time.time() - start_time))
