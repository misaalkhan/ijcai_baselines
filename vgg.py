import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import random
import torch
import torch.nn as nn
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split

BASE_OUTPUT = "."
# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "ijcai_verify/models/aiims_vgg19_again.pt")

df1 = pd.read_csv('child_dataset_train.csv')
df2 = pd.read_csv('child_dataset_valid.csv')
df3 = pd.read_csv('child_dataset_test.csv')
image_paths1 = df1['frontal1'].tolist()
image_paths2 = df2['frontal1'].tolist()
image_paths3 = df3['frontal1'].tolist()
# image_paths_unlabeled = df['frontal1'].tolist()
label1 = df1['target'].tolist()
label2 = df2['target'].tolist()
label3 = df3['target'].tolist()
# print(label1)
# print(type(label1))
class LabeledDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.labels = [int(label) for label in labels]
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[index]

transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize((224,224))
])



dataset_b = LabeledDataset(image_paths1, label1, transform=transform)
dataset_v = LabeledDataset(image_paths2, label2, transform=transform)
dataset_t = LabeledDataset(image_paths3, label3, transform=transforms.ToTensor())




# Set the seed for reproducibility
random.seed(22)

# Split the dataset_b into train, validation, and test datasets
# Calculate the size of each split based on the desired ratio
# train_size = int(len(dataset_b) * 0.6)
# valid_size = int(len(dataset_b)*0.2)
# test_size = len(dataset_b) - train_size-valid_size

# Use the random_split method to divide the dataset into the required splits
# train_dataset,valid_dataset,test_dataset = torch.utils.data.random_split(dataset_b, [train_size,valid_size,test_size])

train_dataloader_b = torch.utils.data.DataLoader(dataset_b, batch_size=32, shuffle=True, num_workers=10)
valid_dataloader_b = torch.utils.data.DataLoader(dataset_v, batch_size=32, shuffle=True, num_workers=8)
test_dataloader_b = torch.utils.data.DataLoader(dataset_t, batch_size=1, shuffle=False, num_workers=0)

# train_size = int(len(dataset) * 0.6)
# test_size = len(dataset) - train_size 

# Use the random_split method to divide the dataset into the required splits
# train_dataset,test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

def make_train_step(model, optimizer, loss_fn):
  def train_step(x,y):
    #make prediction
    yhat = model(x)
    #enter train mode
    model.train()
    #compute loss
    loss = loss_fn(yhat,y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    #optimizer.cleargrads()

    return loss
  return train_step

from torchvision import datasets, models, transforms
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.vgg19(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Enable gradient computation for new layers
for param in model.classifier.parameters():
    param.requires_grad = True

nr_filters = model.classifier[-1].in_features # number of input features of last layer in the classifier
model.classifier[-1] = nn.Linear(nr_filters, 1) # replace last layer with a single output unit

# Move the model to the desired device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# for params in model.parameters():
#   params.requires_grad_ = False

# #add a new final layer
# nr_filters = model.fc.in_features  #number of input features of last layer
# model.fc = nn.Linear(nr_filters, 1)

model = model.to(device)

from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.optim import lr_scheduler

#loss
loss_fn = BCEWithLogitsLoss() #binary cross entropy with sigmoid, so no need to use sigmoid in the model

#optimizer
learning_rate = 0.001
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate) 


#train step
train_step = make_train_step(model, optimizer, loss_fn)

def accuracy(y_pred, y_true):
    y_pred = torch.round(torch.sigmoid(y_pred))
    correct = (y_pred == y_true).float()
    acc = correct.sum() / len(correct)
    return acc
from tqdm import tqdm


losses = []
val_losses = []

epoch_train_losses = []
epoch_test_losses = []
train_accs = []
val_accs = []
n_epochs=1
early_stopping_tolerance = 3
early_stopping_threshold = 0.03

for epoch in range(n_epochs):
  epoch_loss = 0
  epoch_train_acc = 0
  for i ,data in tqdm(enumerate(train_dataloader_b), total = len(train_dataloader_b)): #iterate ove batches
    x_batch , y_batch = data
    x_batch = x_batch.to(device) #move to gpu
    y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
    y_batch = y_batch.to(device) #move to gpu

    yhatrain = model(x_batch)
    loss = train_step(x_batch, y_batch)
    epoch_loss += loss/len(train_dataloader_b)
    losses.append(loss)
    acc = accuracy(yhatrain, y_batch)
    epoch_train_acc += acc / len(train_dataloader_b)
    train_accs.append(acc)
  epoch_train_losses.append(epoch_loss)
#   print('\nEpoch : {}, train loss : {}'.format(epoch+1,epoch_loss))

  #validation doesnt requires gradient
  with torch.no_grad():
    cum_loss = 0
    epoch_test_acc = 0
    for x_batch, y_batch in valid_dataloader_b:
      x_batch = x_batch.to(device)
      y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
      y_batch = y_batch.to(device)

      #model to eval mode
      model.eval()

      yhat = model(x_batch)
      val_loss = loss_fn(yhat,y_batch)
      cum_loss += loss/len(valid_dataloader_b)
      val_losses.append(val_loss.item())
      acc = accuracy(yhat, y_batch)
      epoch_test_acc += acc / len(valid_dataloader_b)
      val_accs.append(acc)


    epoch_test_losses.append(cum_loss)
    print('Epoch : {}, train loss : {}, train acc: {}, val loss : {}, val acc: {}'.format(epoch+1, epoch_loss, epoch_train_acc, cum_loss, epoch_test_acc))
    
    best_loss = min(epoch_test_losses)
    
    #save best model
    if cum_loss <= best_loss:
      best_model_wts = model.state_dict()
    
    #early stopping
    early_stopping_counter = 0
    if cum_loss > best_loss:
      early_stopping_counter +=1

    if (early_stopping_counter == early_stopping_tolerance) or (best_loss <= early_stopping_threshold):
      print("/nTerminating: early stopping")
      break #terminate training
    
#load best model
# model.load_state_dict(best_model_wts)
torch.save(best_model_wts, 'ijcai_verify/models/aiims_vgg19_again.pt')
torch.save(model, MODEL_PATH)

import matplotlib.pyplot as plt

# create the plot
plt.plot(epoch_train_losses, label='Training loss')
plt.plot(epoch_test_losses, label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# save the plot with a specific name
plt.savefig('ijcai_verify/graphs/vgg_16.jpg')

plt.plot(train_accs, label='Training accu')
plt.plot(val_accs, label='Validation accu')
plt.title('Training and Validation accu')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('ijcai_verify/graphs/vgg_16.jpg')


model = torch.load(MODEL_PATH).to(device)

# Define a function for evaluating the model on the test set
def test(model, test_dataloader):
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for x_batch, y_batch in test_dataloader_b:
            x_batch = x_batch.to(device)
            y_batch = y_batch.unsqueeze(1).float().to(device)
            y_pred = model(x_batch)
            test_loss += loss_fn(y_pred, y_batch).item() / len(test_dataloader_b)
            test_acc += accuracy(y_pred, y_batch).item() / len(test_dataloader_b)
    return test_loss, test_acc

# Evaluate the model on the test set
test_loss, test_acc = test(model, test_dataloader_b)

# Print the test results
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")


