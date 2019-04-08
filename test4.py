import csv
from copy import deepcopy
import operator
import time
start_time = time.time()
import numpy as np

 
increment_cost = [19296, 19296, 110592, 110592, 491520, 491520, 491520, \
					 1966080, 1966080, 1966080, 2359296, 2359296, 2359296, 5120]
total_bits = 2*sum(increment_cost)
max_prec = 4
accuracy_mat = [[0] for x in range(14)]
with open('quantized_data.csv') as csvDataFile:
	csvReader = csv.reader(csvDataFile)
	ind = 0
	for row in csvReader:
		accuracy_mat[ind]=list(map(float,row[0:]))
		ind += 1
'''
visited_mat = [0,0,0,0]
for layer in range(13):
	visited_mat = [deepcopy(visited_mat) for x in range(4)]
#print(visted_mat[3][3][3][3][3][3][3][3][3][3][3][3][3][3])
'''
shape = [max_prec]*14
visited_mat = np.zeros(shape=shape, dtype=bool)
final_mat = np.zeros(shape=shape, dtype=bool)
working_mat = np.zeros(shape=shape, dtype=bool)

class QuantScheme:
	def __init__(self):
		 self.weights = [1]*14
		 self.accuracy = 0
		 self.bits = sum(increment_cost)
		 self.final = False
	def increment(self, layer):
		self.weights[layer] += 1
		self.accuracy += accuracy_mat[layer][self.weights[layer]-1] \
						-accuracy_mat[layer][self.weights[layer]-2]
		self.bits += increment_cost[layer]
	def __eq__(self, other) : 
		return self.weights == other.weights

# Initialize list of quantization schemes        
def find_scheme():
	if sum(increment_cost)+min(increment_cost)<=total_bits:
		working_list = []
		final_list = []
		working_list.append(QuantScheme())
		while len(working_list)>0:
			print(len(final_list),len(working_list))
			template_scheme = working_list.pop()
			#trash_list.append(template_scheme)
			x = [weight-1 for weight in template_scheme.weights]
			visited_mat[x[0]][x[1]][x[2]][x[3]][x[4]][x[5]][x[6]][x[7]][x[8]][x[9]][x[10]][x[11]][x[12]][x[13]]=1
			template_scheme.final = True
			for layer in range(14):
				if (increment_cost[layer] + template_scheme.bits <= total_bits) \
					and (template_scheme.weights[layer] < max_prec):
					template_scheme.final = False
					current_scheme = deepcopy(template_scheme)
					current_scheme.increment(layer)
					x = [weight-1 for weight in current_scheme.weights]
					#print(x)
					#print(visited_mat[x[0]][x[1]][x[2]][x[3]][x[4]][x[5]][x[6]][x[7]][x[8]][x[9]][x[10]][x[11]][x[12]][x[13]])
					if final_mat[x[0]][x[1]][x[2]][x[3]][x[4]][x[5]][x[6]][x[7]][x[8]][x[9]][x[10]][x[11]][x[12]][x[13]] != 1 and \
					   working_mat[x[0]][x[1]][x[2]][x[3]][x[4]][x[5]][x[6]][x[7]][x[8]][x[9]][x[10]][x[11]][x[12]][x[13]] != 1 and \
					   visited_mat[x[0]][x[1]][x[2]][x[3]][x[4]][x[5]][x[6]][x[7]][x[8]][x[9]][x[10]][x[11]][x[12]][x[13]] != 1:
						working_list.append(current_scheme)
						working_mat[x[0]][x[1]][x[2]][x[3]][x[4]][x[5]][x[6]][x[7]][x[8]][x[9]][x[10]][x[11]][x[12]][x[13]] = 1
			# Couldn't increment scheme further, pushed final version to list
			if (template_scheme.final) == True:
				final_list.append(template_scheme)
				x = [weight-1 for weight in template_scheme.weights]
				final_mat[x[0]][x[1]][x[2]][x[3]][x[4]][x[5]][x[6]][x[7]][x[8]][x[9]][x[10]][x[11]][x[12]][x[13]] = 1
		final_list.sort(key=operator.attrgetter('accuracy'))
		return final_list[-1]
	elif sum(increment_cost)<=total_bits:
		base = QuantScheme()
		base.final = True
		return base
	else:
		raise Exception('No quantization scheme could be found for the specified number of total bits.')


scheme = find_scheme()
print(scheme.weights)
print("--- %s seconds ---" % (time.time() - start_time))