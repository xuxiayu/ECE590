import torch
import torchvision
import torchvision.transforms as transforms
from modules.quantize import quantize, quantize_grad, QConv2d, QLinear, RangeBN
import torch.nn as nn
import torch.nn.functional as F
import csv
import torch.optim as optim
########################################################################
# Specify Parameters
output_file = 'tim_100k.csv'
max_bits = 100000
########################################################################
# Initialization
conv1_w = 0
conv2_w = 0
fc1_w = 0
fc2_w = 0
fc3_w = 0
transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                  shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                               download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                 shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

########################################################################
def main():
    global conv1_w, conv2_w, fc1_w, fc2_w, fc3_w
    for a in range(1,3):
        conv1_w = a
        for b in range(1,9):
            conv2_w = b
            for c in range(1,9):
                fc1_w = c
                for d in range(1,9):
                    fc2_w = d
                    for e in range(1,9):
                        fc3_w = e
                        totBits = a * 4704 + b * 1600 + c * 48000 + d * 10080 + e * 840
                        if totBits < max_bits:
                            runNet()

def runNet():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.conv1 = QConv2d(3, 6, 5, num_bits_weight = conv1_w)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = QConv2d(6, 16, 5, num_bits_weight = conv2_w)
            self.fc1 = QLinear(16 * 5 * 5, 120, num_bits_weight = fc1_w)
            self.fc2 = QLinear(120, 84, num_bits_weight = fc2_w)
            self.fc3 = QLinear(84, 10, num_bits_weight = fc3_w)

        def update(self):
            self.__init__()

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    print("Starting")
    net = Net()
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    ########################################################################
    # Train the network

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    
    ########################################################################
    #Append CSV
    print('Finished Training')
    fields = ['']*17
    fields[0] = str(max_bits)
    fields[1] = str(conv1_w)
    fields[2] = str(conv2_w)
    fields[3] = str(fc1_w)
    fields[4] = str(fc2_w)
    fields[5] = str(fc3_w)
    fields[6] = "{:.3f}".format(100 * correct / total)
    for i in range(10):
        fields[i+7] = "{:.3f}".format(100 * class_correct[i] / class_total[i])
    outfile = open(output_file, 'a', newline='')
    writer = csv.writer(outfile)
    writer.writerow(fields)
    outfile.close()


if __name__ == '__main__':
    main()