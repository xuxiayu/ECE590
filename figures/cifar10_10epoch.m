bits = 1:8;
layer1 = [53.59 58.61 59.27 60.07 60.57 58.93 61.78 59.91];
layer2 = [56.95 58.79 61.55 61.34 62.91 62.31 60.8 62.64];
layer3 = [52.33 61.3 61.8 61.89 60.71 61.61 61.58 60.34];
layer4 = [58.29 59.37 60.23 60.25 61.3 62.49 60.06 60.41];
layer5 = [49.1 57.23 56.72 60.18 61.35 60.97 60.71 61.71];
figure; hold on
plot(bits,layer1,'-o')
plot(bits,layer2,'-o')
plot(bits, layer3,'-o')
plot(bits, layer4,'-o')
plot(bits, layer5,'-o')
legend('conv1','conv2','fc1','fc2','fc3')
xlabel('Quantization Bits')
ylabel('Overall Accuracy (%)')
title('LeNet Quantization By Layer')
axis([0.5 8.5 48 64])


