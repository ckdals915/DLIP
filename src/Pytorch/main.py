'''
* ***********************************************************************************************
* @author	  ChangMin An
* @Mod		  2022 - 05 - 11
* @brief	  DLIP_CNN Classification Train and Test (efficient_b7 model)
* @Version	Python 3.7, CUDA 11.6(RTX 3060 Laptop), Keras 2.8.0, tensorflow 2.8.0, pytorch 11.3
* ***********************************************************************************************
'''

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
from torchsummary import summary

# from set_parameter_requires_grad import set_parameter_requires_grad
from initialize_model import initialize_model
from train import train
from test import test

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {device} device")
if torch.cuda.is_available(): print(f'Device name: {torch.cuda.get_device_name(0)}') 

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception*]
model_name = "efficientnet"

# Number of classes in the dataset
num_classes = 2

feature_extract = True   # True: only update the reshaped layer params, False: finetune the whole model, 

####################################################################
### add your code for 'efficientnet_b7' in 'initialize_model.py' ###
####################################################################
#model_ft = models.efficientnet_b7(pretrained=True)
#model_ft = models.squeezenet1_0(pretrained=True)
#input_size = 600

model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)  
model_ft = model_ft.to(device)


summary(model_ft, (3,input_size,input_size))

print(model_ft)

# Top level data directory. Here we assume the format of the directory conforms 
#   to the ImageFolder structure
data_dir = "data/kagglecatsanddogs/PetImages"

# Data augmentation and normalization for training
# Just normalization for validation
transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

training_data = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), transform = transform['train'])
test_data = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'), transform = transform['test'])

classes = ['Cat', 'Dog']
print(f"train dataset length = {len(training_data)}")
print(f"test  dataset length = {len(test_data)}")


# Batch size for training
batch_size = 16

train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

for X, y in test_dataloader:
  print(f"Shape of X [N, C, H, W]: {X.shape} {y.dtype}")
  print(f"Shape of y: {y.shape} {y.dtype}")
  break


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_ft.parameters(), lr = 0.001, momentum=0.9,weight_decay=5e-4)

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model_ft, loss_fn, optimizer, device, 15)
    test(test_dataloader, model_ft, loss_fn, device)
print("Done!")

print("SAVE DATA")
torch.save(model_ft, f"{model_name}_ft(kagglecatsanddogs)_epochs{epochs}.pth")
print("SAVE DATA Done!")

dataiter = iter(test_dataloader)
images, labels = dataiter.next()

images = images.to(device)
labels = labels.to(device)
pred = model_ft(images)
predicted=pred.argmax(1)

figure = plt.figure()
num_of_images = min(batch_size, 9)

for index in range(num_of_images):
  plt.subplot(3, 3, index+1)
  plt.axis('off')
  plt.title(f"Ground Truth: {classes[labels[index]]}")
  plt.title(f"{classes[predicted[index].item()]} (true:{classes[labels[index]]})")
  plt.imshow(np.transpose((images[index] * 0.224 + 0.456).cpu().numpy().squeeze(), (1, 2, 0)))


heatmap = pd.DataFrame(data=0, index=classes, columns=classes)
with torch.no_grad():
  for images, labels in test_dataloader:
    images, labels = images.to(device), labels.to(device)
    outputs = model_ft(images)
    _, predicted = torch.max(outputs, 1)
    c = (predicted == labels).squeeze()
    for i in range(len(labels)):
      true_label = labels[i].item()
      predicted_label = predicted[i].item()
      heatmap.iloc[true_label, predicted_label] += 1
_, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(heatmap, annot=True, fmt="d", cmap="YlGnBu")
plt.show()

