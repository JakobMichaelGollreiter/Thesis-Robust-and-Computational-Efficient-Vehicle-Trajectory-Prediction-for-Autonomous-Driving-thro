# imports
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.utils.data.dataset import random_split
import numpy as np
from torchinfo import summary
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import datetime
import os

# print(f'is cuda available: {torch.cuda.is_available()}')

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

# Data Augmentation 
my_transforms = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.0)
    # transforms.ToTensor()
])

class CustomDataset(Dataset): # create Dataset class
    # def __init__(self, csv_file):
    #     self.data = self.load_csv(csv_file)
    def __init__(self, csv_file, my_transforms):
        self.data = self.load_csv(csv_file)
        self.transform = my_transforms
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        filename = str(row['\ufefffilename'])
        cr_id = int(row['cr_id'])
        time_begin = int(row['time_begin'])
        mapInView = eval(row['mapInView'])
        socialInformationOfCarsInView = eval(row['socialInformationOfCarsInView'])
        groundTruthTrajectory = eval(row['groundTruthTrajectory'])
        
        sample = {
            'filename' : filename,
            'cr_id': cr_id,
            'time_begin': time_begin,
            'mapInView': torch.tensor(mapInView),
            'socialInformationOfCarsInView': torch.tensor(socialInformationOfCarsInView),
            'groundTruthTrajectory': torch.tensor(groundTruthTrajectory)
        }
        # sample['mapInView'] = my_transforms(sample['mapInView'])

        return sample


    def load_csv(self, csv_file):
        # Load the CSV file and return a list of dictionaries
        # Each dictionary represents a row with column names as keys
        # You can use any CSV parsing library or write your own logic
        # Here, we'll use the 'csv' module from the Python standard library

        import csv

        data = []
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                print(row)
                data.append(row)

        return data
current_directory = os.getcwd()  # Get the current working directory
#workingDirectory = os.path.abspath(os.path.join(current_directory, ".."))

databasePath = 'imageTrain2'
datasetPath = os.path.join(current_directory,'..','db39', databasePath, 'combined.csv') # path to local dataset
# datasetPath = os.path.join(current_directory,'db39', databasePath, 'combined.csv') # path to dataset on GPU server
print(datasetPath)


dataset = CustomDataset(datasetPath, my_transforms) # create dataset cobined data with data augmenatiation 
sample = dataset.__getitem__(0)
input_sizeX = sample['mapInView'].shape[0]
input_sizeY = sample['mapInView'].shape[1]

train_ratio = 0.88
val_ratio = 0.1
test_ratio = 0.02

num_samples = len(dataset)
train_size = int(train_ratio * num_samples)
val_size = int(val_ratio * num_samples)
test_size = num_samples - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
num_data_items = len(dataset)
print("Number of data items:", num_data_items)

batch_size = 32
test_batch_size = 1

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

##########################
modelName = 'LinearModelLatentSpace64x1x1'
##########################

class AE(torch.nn.Module): # make Autoencoder class Linear
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_sizeX * input_sizeY, 256), # Building an linear encoder with Linear
            torch.nn.ReLU(), # layer followed by Relu activation function
            # torch.nn.Linear(512, 256),
            # torch.nn.ReLU(), # layer followed by Relu activation function
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(), # layer followed by Relu activation function
            torch.nn.Linear(128, 64),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            # torch.nn.Linear(256, 512),
            # torch.nn.ReLU(),
            torch.nn.Linear(256, input_sizeX * input_sizeY),
            torch.nn.Sigmoid() # The Sigmoid activation function outputs the value between 0 and 1
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
model = AE().to(device) # make linaer model     
# summary(model, (1, input_sizeX * input_sizeY))

# print(f"Is model on GPU: {model.is_cuda}")
# print(f"Is model on GPU: {next(model.parameters()).is_cuda}")
print('\n')


criterion = torch.nn.MSELoss()
lr = 3e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epochs = 200  # Set your desired number of epochs
print(f'Training {modelName} for {epochs} epochs:')
outputs = []
losses = []
epochLoss = []
validationLoss = []
overallValidationLoss = 0
validationLossHistory = []
for epoch in range(epochs):
    for batch in train_loader:
        images = batch['mapInView'].view(-1, input_sizeX * input_sizeY).to(device)

        reconstructed = model(images).to(device)  # Output of Autoencoder
        loss = criterion(reconstructed, images)  # Calculating the loss function

        optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update the parameters

        epochLoss.append(loss.item())  # Storing the losses in a list for plotting
    losses.append(sum(epochLoss)/len(epochLoss)) # make better
    for batch in validation_loader:
        images = batch['mapInView'].view(-1, input_sizeX * input_sizeY).to(device)
        reconstructed = model(images).to(device)  # Output of Autoencoder
        loss = criterion(reconstructed, images)  # Calculating the loss function
        validationLoss.append(loss.item())  # Storing the losses in a list for plotting
    overallValidationLoss = sum(validationLoss)/len(validationLoss) 
    validationLossHistory.append(overallValidationLoss)
    epochLoss = []

    print('   Epoch: [{:2d}|{:2d}]   Train Loss: {:.7f}   Validation Loss: {:.7f}'.format(epoch + 1, epochs, losses[-1], overallValidationLoss))
    outputs.append((epoch + 1, images, reconstructed)) # Store outputs for visualization if needed

print('DONE TRAINING')


now = datetime.datetime.now()
timestamp = now.strftime("%Y_%m_%d_%H:%M:%S")
file_name = f'{datasetPath}_{modelName}_Epochs{epochs:}_ValidationLoss:{float(validationLossHistory[-1]):.6f}_lr:{lr:.4f}__{timestamp}.pth'
# Get the current directory path
current_directory = os.getcwd()  # Get the current working directory
# current_directory = os.path.abspath(os.path.join(current_directory, ".."))

# Concatenate the current directory path with the file name
file_path = os.path.join(current_directory, file_name)

# Save the model's state dictionary to the specified file path
# torch.save(model.state_dict(), file_path)
torch.save(model, file_path) # save the whole model
print(f'saved model: {file_name}')
