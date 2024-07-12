from astrosite_dataset import ClassificationAstrositeDataset, TrackingAstrositeDataset, SpectrogramDataset
from classification_network import LeNet5
import os
import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

dataset_path = '../filtered_dataset/recordings'
assert os.path.isdir(dataset_path)

target_list = ['50574', '47851', '37951', '39533', '43751', '32711', '27831', '45465',
       '46826', '42942'] #, '42741', '41471', '43873', '40982', '41725', '43874', 
       #'27711', '40892', '50005', '44637']

train_dataset = SpectrogramDataset(dataset_path, split=target_list, test=False)
test_dataset = SpectrogramDataset(dataset_path, split=target_list, test=True)


# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using ", device)
batch_size = 80
num_classes = len(target_list)
net = LeNet5(num_classes).to(device)
model_filename = "./networks/classification_easy.pt"
learning_rate = 0.001
num_epochs = 10
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

print(train_dataset[0][0].shape)
print(len(train_dataset))

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True,
                                           num_workers = 4)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = batch_size,
                                           shuffle = True,
                                           num_workers = 4)

total_step_train = len(train_loader)
total_step_test = len(test_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        #Forward pass
        outputs = net(images)
        loss = cost(outputs, labels)
        	
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
        		           .format(epoch+1, num_epochs, i+1, total_step_train, loss.item()))
        sys.stdout.flush()
    for j, (images, labels) in enumerate(test_loader):  
        images = images.to(device)
        labels = labels.to(device)
        
        #Forward pass
        outputs = net(images)
        loss = cost(outputs, labels)
        	
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print('Test: epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
        		           .format(epoch+1, num_epochs, j+1, total_step_test, loss.item()))
        sys.stdout.flush()
    torch.save(net.state_dict(), model_filename)

