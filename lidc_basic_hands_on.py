import time
import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import imageio.v3 as iio
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import WeightedRandomSampler
import pydicom

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]),
}

# Specify the directory path
directory = 'C:/Users/Rohan/Lung Cancer Detection/manifest-1600709154662/LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-NA-NA-30178/3000566.000000-NA-03192'

# Initialize an empty list to store the images
images = []

# Iterate over the files in the directory
for file in Path(directory).iterdir():
    # Check if the file is a DICOM file
    if file.is_file() and file.suffix == '.dcm':
        img = iio.imread(file, plugin='DICOM')

        images.append(img)
# Now you can access the images in the list
print(len(images))  # Number of images in the list
print(images[0].shape)  # Shape of the first image
##Accessing the dataframe
dataframe = pd.read_csv("C:/Users/Rohan/Lung Cancer Detection/manifest-1600709154662/metadata.csv")
print(dataframe)
##To preview Image
file_path = "C:/Users/Rohan/Lung Cancer Detection/manifest-1600709154662/LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-NA-NA-30178/3000566.000000-NA-03192/1-001.dcm"  # Replace 'username' and 'your_dcm_file.dcm' with your actual username and dcm file name
ds = pydicom.dcmread(file_path)

# Rescale the image data
rescale_intercept = ds.RescaleIntercept
rescale_slope = ds.RescaleSlope
pixel_array = ds.pixel_array * rescale_slope + rescale_intercept

# Convert the image data to a NumPy array
image = np.array(pixel_array, dtype=float)

print(image)
plt.imshow(image)

###Data preprocessing
data_dir = "C:/Users/Rohan/Lung Cancer Detection/manifest-1600709154662/LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-NA-NA-30178/3000566.000000-NA-03192" # Set the directory for the data
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['test', 'train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                              shuffle=True, num_workers=2)
               for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['test', 'train', 'val']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(class_names)


### Visualization  Function


def imshow(inp, title):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()


# Get a batch of testing data
inputs, classes = next(iter(dataloaders['test']))
# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])


def plot(val_loss, train_loss, typ):
    plt.title("{} after epoch: {}".format(typ, len(train_loss)))
    plt.xlabel("Epoch")
    plt.ylabel(typ)
    plt.plot(list(range(len(train_loss))), train_loss, color="r", label="Train " + typ)
    plt.plot(list(range(len(val_loss))), val_loss, color="b", label="Validation " + typ)
    plt.legend()
    plt.savefig(os.path.join(data_dir, typ + ".png"))
    plt.close()


##Model Training Function
def train_model(model, criterion, optimizer, scheduler, num_epochs=25, model_name="kaggle", train_loss_gph=None,
                train_acc_gph=None, val_loss_gph=None, val_acc_gph=None):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs,
                                         1)  #was (outputs,1) for non-inception and (outputs.data,1) for inception
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                train_loss_gph.append(epoch_loss)
                train_acc_gph.append(epoch_acc)
            if phase == 'val':
                val_loss_gph.append(epoch_loss)
                val_acc_gph.append(epoch_acc)
            # plot(val_loss_gph,train_loss_gph, "Loss")
            #  plot(val_acc_gph,train_acc_gph, "Accuracy")
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model, data_dir + "/" + model_name + ".h5")
                print('==>Model Saved')
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
# ResNet 152

model = models.resnet152(pretrained = True)
#num_ftrs = model.classifier[0].in_features
num_ftrs = model.fc.in_features  ##for googlenet, resnet18
#num_ftrs = model.classifier.in_features  ## for densenet169
print("Number of features: "+str(num_ftrs))
#model.classifier = nn.Linear(num_ftrs, num_classes) ## for vgg19
model.fc = nn.Linear(num_ftrs, num_classes)  ##for googlenet, resnet18
#model.classifier = nn.Linear(num_ftrs, num_classes) ## for densenet169
model = model.to(device)
criterion = nn.CrossEntropyLoss( weight = torch.tensor([1, 4.7]).to(device))
# Observe that all parameters are being optimized
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# StepLR Decays the learning rate of each parameter group by gamma every step_size epochs
# Decay LR by a factor of 0.1 every 7 epochs
# Learning rate scheduling should be applied after optimizer’s update
# e.g., you should write your code this way:
# for epoch in range(100):
#     train(...)
#     validate(...)
#     scheduler.step()
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma=0.1)
model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=10, model_name = "resnet152")
# Getting Proba distribution
print("\nGetting the Probability Distribution")
trainloader=torch.utils.data.DataLoader(image_datasets['train'],batch_size=1)
testloader=torch.utils.data.DataLoader(image_datasets['test'],batch_size=1)
model=model.eval()
correct = 0
total = 0
import csv
import numpy as np  # Importing NumPy for numerical operations

f = open(data_dir+"/resnet152_train.csv",'w+',newline = '')
writer = csv.writer(f)
saving = []
with torch.no_grad():
      num = 0
      temp_array = np.zeros((len(trainloader),num_classes))
      for i,data in enumerate(trainloader):
          images, labels = data
          sample_fname, _ = trainloader.dataset.samples[i]
          labels=labels.cuda()
          outputs = model(images.cuda())
          _, predicted = torch.max(outputs, 1)
          total += labels.size(0)
          correct += (predicted == labels.cuda()).sum().item()
          prob = torch.nn.functional.softmax(outputs, dim=1)
          saving.append(sample_fname.split('/')[-1])
          temp_array[num] = np.asarray(prob[0].tolist()[0:num_classes])
          num+=1
print("Train Accuracy = ",100*correct/total)
for i in range(len(trainloader)):
  k = temp_array[i].tolist()
  k.append(saving[i])
  writer.writerow(k)
f.close()
f = open(data_dir+"/train_labels.csv",'w+',newline = '')
writer = csv.writer(f)
for i,data in enumerate(trainloader):
  _, labels = data
  sample_fname, _ = trainloader.dataset.samples[i]
  sample = sample_fname.split('/')[-1]
  lab = labels.tolist()[0]
  writer.writerow([sample,lab])
f.close()


#%%
#Test Probabilities
f = open(data_dir+"/resnet152_test.csv",'w+',newline = '')
writer = csv.writer(f)
saving = []
with torch.no_grad():
      num = 0
      temp_array = np.zeros((len(testloader),num_classes))
      for i,data in enumerate(testloader):
          images, labels = data
          sample_fname, _ = testloader.dataset.samples[i]
          labels=labels.cuda()
          outputs = model(images.cuda())
          _, predicted = torch.max(outputs, 1)
          total += labels.size(0)
          correct += (predicted == labels.cuda()).sum().item()
          prob = torch.nn.functional.softmax(outputs, dim=1)
          saving.append(sample_fname.split('/')[-1])
          temp_array[num] = np.asarray(prob[0].tolist()[0:num_classes])
          num+=1
print("Test Accuracy = ",100*correct/total)
for i in range(len(testloader)):
  k = temp_array[i].tolist()
  k.append(saving[i])
  writer.writerow(k)
f.close()
f = open(data_dir+"/test_labels.csv",'w+',newline = '')
writer = csv.writer(f)
for i,data in enumerate(testloader):
  _, labels = data
  sample_fname, _ = testloader.dataset.samples[i]
  sample = sample_fname.split('/')[-1]
  lab = labels.tolist()[0]
  writer.writerow([sample,lab])
f.close()

# Ensemble Learning Analysis
