'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import csv
from models import *
from utils import progress_bar
import itertools
import time
start_time = time.time()

increment_cost = [19296, 19296, 110592, 110592, 491520, 491520, 491520, \
					 1966080, 1966080, 1966080, 2359296, 2359296, 2359296, 5120]

torch.set_num_threads(4)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--max_prec', default = 4, type=int, help='max weight precision')
parser.add_argument('--bits', default = 2*sum(increment_cost), type=int, help='total available bits')
parser.add_argument('--fn', default = "placeholder", help='filename')
args = parser.parse_args()
output_file = args.fn + ".csv"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128*2, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

##########################################################################################
##########################################################################################

# Training
def train(epoch):
	print('\nEpoch: %d' % epoch)
	net.train()
	train_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(trainloader):
		inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()

		progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
	global best_acc, field, weight_bits, net, acc, used_bits
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(testloader):
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = net(inputs)
			loss = criterion(outputs, targets)

			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()

			progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
				% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
	acc = 100.*correct/total
	field.append(acc)
	print(field)
	if acc > best_acc:
		best_acc = acc

def create_checkpoint():
	print('==> Saving..')
	state = {
		'net': net.state_dict(),
		'acc': acc,
		'epoch': epoch,
		'weight_bits' : weight_bits,
		'field' : field
	}
	if not os.path.isdir('checkpoint'):
		os.mkdir('checkpoint')
	torch.save(state, './checkpoint/ckpt.t7')

def load_checkpoint():
	global best_acc, weight_bits, field, start_epoch, used_bits
	print('==> Resuming from checkpoint..')
	assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
	checkpoint = torch.load('./checkpoint/ckpt.t7')
	best_acc = checkpoint['acc']
	start_epoch = checkpoint['epoch']+1
	weight_bits = checkpoint['weight_bits']
	net = QVGG('VGG16', weight_bits)
	net.load_state_dict(checkpoint['net'])
	field = checkpoint['field']
	for x in range(len(weight_bits)):
		used_bits += weight_bits[x]*increment_cost[x]
	#if used bits > total bits, throw error
	return net

# Updating
def update_net(net,weight_bits):
	existing_dict = net.state_dict()
	new_net = QVGG('VGG16', weight_bits)
	new_dict = new_net.state_dict()
	new_dict.update(existing_dict)
	new_net.load_state_dict(new_dict)
	return net

def update_fields():
	global field
	field[0] = total_bits
	field[1] = used_bits
	for col in range(2,2+len(weight_bits)):
		field[col]=weight_bits[col-2]

def write_csv():
	outfile = open(output_file, 'a', newline='')
	writer = csv.writer(outfile)
	writer.writerow(field)
	outfile.close()


# Initialize list of quantization schemes        
def find_scheme():
	print('==> Finding quantization scheme..')
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
			#utilization threshold to save time
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

##########################################################################################
##########################################################################################
# Initialization
best_acc = 0  # best test accuracy
acc = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
used_bits = 0
weight_bits = [0]*len(increment_cost)
total_bits = args.bits
max_precision = args.max_prec
field = [0]*(2+len(weight_bits))

# read in sensitivity analysis
accuracy_mat = [[0] for x in range(14)]
with open('quantized_data.csv') as csvDataFile:
	csvReader = csv.reader(csvDataFile)
	ind = 0
	for row in csvReader:
		accuracy_mat[ind]=list(map(float,row[0:]))
		ind += 1

# Initialize net
if args.resume:
	net = load_checkpoint()
else:
	start_time = time.time()
	weight_bits = find_scheme()
	print(scheme)
	print("--- %s seconds ---" % (time.time() - start_time))
	used_bits = sum([a*b for a,b in zip(weight_bits,increment_cost)])
	print('==> Building model..')
	net = QVGG('VGG16', weight_bits)
net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

# Initialize csv
update_fields()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

for epoch in range(start_epoch, start_epoch+10):
	train(epoch)
	test(epoch)
	create_checkpoint()
	write_csv()

	