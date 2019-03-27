import csv
import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from custom_modules.quantize import quantize, quantize_grad, QConv2d, QLinear, RangeBN
import torchvision

########################################################################
# Initialization
output_file = 'tim_v2.csv'
conv1_w = 7
conv2_w = 8
conv3_w = 8
conv4_w = 8
conv5_w = 8
conv6_w = 8
conv7_w = 8
conv8_w = 8
conv9_w = 8
conv10_w = 8
conv11_w = 8
conv12_w = 8
conv13_w = 8
fc1_w = 8
fc2_w = 8
fc3_w = 8
#follow instructions here to download and format ImageNet data: https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset 
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])
valdir = '../data/ILSVRC2012/small_val' 
val_loader = torch.utils.data.DataLoader(
		datasets.ImageFolder(valdir, transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			normalize,
		])),
		batch_size=4, shuffle=False,
		num_workers=8, pin_memory=True)
########################################################################
def main():
	global conv1_w,conv2_w,conv3_w,conv4_w,conv5_w,conv6_w,conv7_w,conv8_w,conv9_w,conv10_w,conv11_w,conv12_w,conv13_w, fc1_w, fc2_w, fc3_w
	vgg16 = models.vgg16(pretrained=True)
	pretrained_dict = vgg16.state_dict()
	
	for conv1_w in range(8,9):
		qvgg16 = QVGG16() 
		model_dict = qvgg16.state_dict()
		
		# update and load
		
		model_dict.update(pretrained_dict)
		qvgg16.load_state_dict(model_dict)
		
		# test accuracy

		qvgg16.eval()
		#vgg16.eval()

		correct = 0
		total = 0
		with torch.no_grad():
			for data in val_loader:
				images, labels = data
				outputs = qvgg16(images)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
		print('Accuracy of the network on the test images: %d %%' % (
			100 * correct / total))
		fields = ['']*17
		fields[0] = str(conv1_w)
		fields[1] = str(conv2_w)
		fields[2] = str(conv3_w)
		fields[3] = str(conv4_w)
		fields[4] = str(conv5_w)
		fields[5] = str(conv6_w)
		fields[6] = str(conv7_w)
		fields[7] = str(conv8_w)
		fields[8] = str(conv9_w)
		fields[9] = str(conv10_w)
		fields[10] = str(conv11_w)
		fields[11] = str(conv12_w)
		fields[12] = str(conv13_w)
		fields[13] = str(fc1_w)
		fields[14] = str(fc2_w)
		fields[15] = str(fc3_w)
		fields[16] = "{:.3f}".format(100 * correct / total)
		outfile = open(output_file, 'a', newline='')
		writer = csv.writer(outfile)
		writer.writerow(fields)
		outfile.close()


########################################################################
# Quantized VGG Model
class QVGG16(nn.Module):
	def __init__(self):
		super(QVGG16, self).__init__()

		self.features = nn.Sequential(
			QConv2d(3, 64, 3, 1, 1, num_bits_weight = conv1_w), 							#(0)
			nn.ReLU(inplace=True),															#(1)
			QConv2d(64, 64, 3, 1, 1, num_bits_weight = conv2_w),							#(2)
			nn.ReLU(inplace=True),															#(3)
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),	#(4)
			QConv2d(64, 128, 3, 1, 1, num_bits_weight = conv3_w),							#(5)
			nn.ReLU(inplace=True),															#(6)
			QConv2d(128, 128, 3, 1, 1, num_bits_weight = conv4_w),							#(7)
			nn.ReLU(inplace=True),															#(8)
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),	#(9)
			QConv2d(128, 256, 3, 1, 1, num_bits_weight = conv5_w),							#(10)
			nn.ReLU(inplace=True),															#(11)
			QConv2d(256, 256, 3, 1, 1, num_bits_weight = conv6_w),							#(12)
			nn.ReLU(inplace=True),															#(13)
			QConv2d(256, 256, 3, 1, 1, num_bits_weight = conv7_w),							#(14)
			nn.ReLU(inplace=True),															#(15)
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),	#(16)
			QConv2d(256, 512, 3, 1, 1, num_bits_weight = conv8_w),							#(17)
			nn.ReLU(inplace=True),															#(18)
			QConv2d(512, 512, 3, 1, 1, num_bits_weight = conv9_w),							#(19)
			nn.ReLU(inplace=True),															#(20)
			QConv2d(512, 512, 3, 1, 1, num_bits_weight = conv10_w),							#(21)
			nn.ReLU(inplace=True),															#(22)
			nn. MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),	#(23)
			QConv2d(512, 512, 3, 1, 1, num_bits_weight = conv11_w),							#(24)
			nn.ReLU(inplace=True),															#(25)
			QConv2d(512, 512, 3, 1, 1, num_bits_weight = conv12_w),							#(26)
			nn.ReLU(inplace=True),															#(27)
			QConv2d(512, 512, 3, 1, 1, num_bits_weight = conv13_w),							#(28)
			nn.ReLU(inplace=True),															#(29)
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)	#(30)
		 )
		self.avgpool = nn.AdaptiveAvgPool2d(7)
		self.classifier = nn.Sequential(
			QLinear(25088, 4096, num_bits_weight = fc1_w),	#(0)
			nn.ReLU(inplace=True),							#(1)
			nn.Dropout(p=0.5),								#(2)
			QLinear(4096, 4096, num_bits_weight = fc2_w),	#(3)
			nn.ReLU(inplace=True),							#(4)
			nn.Dropout(p=0.5),								#(5)
			QLinear(4096, 1000, num_bits_weight = fc3_w)	#(6)
			)

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x
			

if __name__ == '__main__':
	main()