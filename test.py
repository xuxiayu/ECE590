import csv

weight_bits = [1,1,1,1,1,1]
increment_cost = [10,1,1,1,1,1]
epoch = 1
total_bits = 20
used_bits = 0
for x in range(len(increment_cost)):
	used_bits += weight_bits[x]*increment_cost[x]

accuracy_mat = [[0]*4 for x in range(6)]
with open('cifar10_data.csv') as csvDataFile:
	csvReader = csv.reader(csvDataFile)
	for row in csvReader:
		chunk = int(row[0])
		prec =  int(row[1])-1
		accuracy_mat[chunk][prec]=list(map(float,row[2:]))


#flag for max utilization
#cost function options
if epoch<100 and min(weight_bits)<4:
	sensitivity = [-float('Inf')]*6
	for chunk in range(len(weight_bits)):
		#if precion is not maxed and bits are available to increment
		if weight_bits[chunk]<4 and (total_bits-used_bits)>=increment_cost[chunk]:
			sensitivity[chunk]=(accuracy_mat[chunk][weight_bits[chunk]][epoch]-accuracy_mat[chunk][weight_bits[chunk]-1][epoch])/increment_cost[chunk]
	max_improvement = max(sensitivity)
	max_index = sensitivity.index(max_improvement)
	if max_improvement > -float('Inf'): #negative infinity
		weight_bits[max_index] += 1
		used_bits += increment_cost[max_index]
	print(weight_bits)
