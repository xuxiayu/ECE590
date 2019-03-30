import csv
 
accuracy_mat = [[0]*4]*6
with open('cifar10_data.csv') as csvDataFile:
	csvReader = csv.reader(csvDataFile)
	for row in csvReader:
		chunk = int(row[0])
		prec =  int(row[1])-1
		accuracy_mat[chunk][prec]=list(map(float,row[2:]))
		print(accuracy_mat[chunk][prec])