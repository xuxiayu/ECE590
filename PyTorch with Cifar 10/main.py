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


torch.set_num_threads(4)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
#parser.add_argument('--layer_chunk', default = 0, type=int, help='layer chunk')
parser.add_argument('--bits', default = 29431168, type=int, help='total available bits')
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
	start_epoch = checkpoint['epoch']
	weight_bits = checkpoint['weight_bits']
	net = QVGG('VGG16', weight_bits)
	net.load_state_dict(checkpoint['net'])
	field = checkpoint['field']
	for x in range(len(increment_cost)):
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
	for col in range(2,8):
		field[col]=weight_bits[col-2]

def write_csv():
	outfile = open(output_file, 'a', newline='')
	writer = csv.writer(outfile)
	writer.writerow(field)
	outfile.close()

def increment_bits(weight_bits):
	global used_bits
	if epoch<100 and min(weight_bits)<4 and min(increment_cost)<(total_bits-used_bits):
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
	return(weight_bits)

##########################################################################################
##########################################################################################
# Initialization
best_acc = 0  # best test accuracy
acc = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
increment_cost = [38720-2*64,221440-2*128,1475328-3*256,5899776-512*3,7079424-512*3,512*10]
weight_bits = [1,1,1,1,1,1]
total_bits = args.bits
field = [0]*8
used_bits = 0
# Initialize net
if args.resume:
	net = load_checkpoint()
else:
	print('==> Building model..')
	net = QVGG('VGG16', weight_bits)
	for x in range(len(increment_cost)):
		used_bits += weight_bits[x]*increment_cost[x]
net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

# Initialize csv
update_fields()

# read in sensitivity analysis
# accuracy_mat[chunk][bits-1][epoch]
accuracy_mat = [[0]*4 for x in range(6)]
with open('../cifar10_data.csv') as csvDataFile:
	csvReader = csv.reader(csvDataFile)
	for row in csvReader:
		chunk = int(row[0])
		prec =  int(row[1])-1
		accuracy_mat[chunk][prec]=list(map(float,row[2:]))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

for epoch in range(start_epoch, start_epoch+10):
	create_checkpoint()
	train(epoch)
	test(epoch)
	# try to update precision if accuracy does not increase by threshold
	if acc < best_acc*1.05:
		print('==> Updating precision..')
		# write to file
		write_csv()
		# update weight bits
		weight_bits = increment_bits(weight_bits)
		update_fields()
		# update net
		net = update_net(net,weight_bits)