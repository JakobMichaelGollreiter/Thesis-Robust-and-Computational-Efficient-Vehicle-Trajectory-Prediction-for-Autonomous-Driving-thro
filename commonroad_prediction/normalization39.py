# imports
from torch.utils.data import Dataset, DataLoader


import matplotlib.pyplot as plt
from torch.utils.data.dataset import random_split
import numpy as np
from torchinfo import summary
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.nn.functional import normalize

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import datetime
import random
import os
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset): # create Dataset class
    def __init__(self, csv_file):
        self.data = self.load_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        filename = str(row['\ufefffilename'])
        cr_id = int(row['cr_id'])
        egoVelocity = float(row['egoVelocity'])
        time_begin = int(row['time_begin'])
        mapInView = eval(row['mapInView'])
        socialInformationOfCarsInView = eval(row['socialInformationOfCarsInView'])
        groundTruthTrajectory = eval(row['groundTruthTrajectory'])
        modGroundTruthTrajectory = [[entry[0], entry[1]] for entry in groundTruthTrajectory]
        

        modGroundTruthTrajectory = torch.tensor([[entry[0], entry[1]] for entry in groundTruthTrajectory]).view(30,2) # ignore the last entry of the ground truth trajectory as this is just a number indicating how the trajectory was generated 
 
        
        sample = {
            'filename' : filename,
            'cr_id': cr_id,
            'time_begin': time_begin,
            'egoVelocity': egoVelocity,
            'mapInView': torch.tensor(mapInView),
            'socialInformationOfCarsInView': torch.tensor(socialInformationOfCarsInView),
            'groundTruthTrajectory': torch.tensor(groundTruthTrajectory),
            'modGroundTruthTrajectory': torch.tensor(modGroundTruthTrajectory.float())
        }
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
workingDirectory = os.path.abspath(os.path.join(current_directory, ".."))

dataBase = 'predictionModelServer/db39/mega'
# datasetPath = os.path.join(dataBase, 'combined.csv')
datasetPath = os.path.join(dataBase, 'cleanedCombine.csv')
datasetPath = os.path.join(workingDirectory, datasetPath)
dataset = CustomDataset(datasetPath) # create dataset cobined data
sample = dataset.__getitem__(0)
input_sizeX = sample['mapInView'].shape[0] 
input_sizeY = sample['mapInView'].shape[1]

# Define the proportions for train, validation, and test sets (e.g., 70%, 15%, 15%)
# train_ratio = 0.88
# val_ratio = 0.1
# test_ratio = 0.02

# Calculate the number of samples for each set
num_samples = len(dataset)
#train_size = int(train_ratio * num_samples)
#val_size = int(val_ratio * num_samples)
# test_size = num_samples - train_size - val_size

# Use random_split to split the dataset
#train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_dataset = dataset

num_data_items = len(dataset)
print("Number of data items:", num_data_items)

##########################
modelName = 'CNNModelFlatten64x1x1' # Trying to reduce the output dimensions of the encoder
##########################
encoded_space_dim = 64
outChannels = 4

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, outChannels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(outChannels, outChannels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(outChannels, 2*outChannels, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(2*outChannels, 4*outChannels, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(4*outChannels, 8*outChannels, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(8*outChannels, 16*outChannels, 3, padding=1, stride = 2),
            nn.ReLU()
        )
        
        ### Flatten layer 
        self.flatten = nn.Flatten(start_dim=1)

        ### Linear section 
        self.encoder_lin = nn.Sequential(
            nn.Linear(64*3*3, 128),
            nn.ReLU(),
            nn.Linear(128, encoded_space_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.view(-1,1,39,39)
        output = self.encoder_cnn(x)
        # print(output.shape)
        output = self.flatten(output)
        output = self.encoder_lin(output)
        return output


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim,128),
            nn.ReLU(),
            nn.Linear(128, 64*3*3)
        )

        ### Unflatten layer 
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64,3,3))
        
        ### convolutional Section
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(16*outChannels, 8*outChannels, 3, padding=1, stride = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(8*outChannels, 4*outChannels, 3, padding=1, stride = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(4*outChannels, 2*outChannels, 3, padding=1, stride = 2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(2*outChannels, outChannels, 3, padding=1, stride = 2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(outChannels, outChannels, 4, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(outChannels, 1, 5, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.view(-1,encoded_space_dim)
        output = self.decoder_lin(x)
        output = self.unflatten(output)
        output = self.decoder_cnn(output)
        return output


class ConvAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, x):
        x = x.view(-1,1,39,39)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
encoder = Encoder()
decoder = Decoder()
file_path = os.path.join(workingDirectory,'predictionModelServer/db39/imageTrain2/combined.csv_CNNModelFlatten64x1x1CombinedTry_Epochs200_ValidationLoss:0.001812_lr:0.0003__2023_07_19_18:02:20.pth')
learntCNN = ConvAutoencoder(encoder, decoder)
learntCNN = torch.load(file_path, map_location=torch.device('cpu')).to(device)
learntCNN.eval()

train_dataset.__getitem__(10)

# train model linear
batch_size = 1  # Set your desired batch size
# test_batch_size = 1 # ToDo test batch size. Here should be no batch

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Create data loaders for the training set
#validation_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Create data loaders for the training set
#test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True) # Create data loaders for the test set

sumGroundTruth = torch.zeros(60).to(device)
sumEgoVelocity = torch.zeros(1).to(device)
sumSocialInformationOfCarsInView = torch.zeros(80).to(device)
sumEncidedMap = torch.zeros(64).to(device)
counter = 0
badData = 0
skip = 0
length = int(len(train_dataset)*0.5) 
# Use tqdm to wrap your loop for a progress bar
for i in tqdm(range(length), desc='Processing data', unit='samples'):
    # get sample with the right map and velocity
    sample = train_dataset.__getitem__(i)
    modGroundTruthTrajectory = sample['modGroundTruthTrajectory'].view(60).to(device)
    socialInformationOfCarsInView = sample['socialInformationOfCarsInView'].view(80).to(device)
    egoVelocity = torch.tensor(sample['egoVelocity']).to(device)
    images = sample['mapInView'].view(-1, input_sizeX, input_sizeY).to(device)
    images = images.unsqueeze(1)
    encodedMap = learntCNN.encoder.forward(images)  # Output of encoder dimensionality reduction

    sumGroundTruth = torch.add(sumGroundTruth, torch.abs(modGroundTruthTrajectory))
    sumSocialInformationOfCarsInView = torch.add(sumSocialInformationOfCarsInView, torch.abs(socialInformationOfCarsInView))
    sumEncidedMap = torch.add(sumEncidedMap, torch.abs(encodedMap))
    sumEgoVelocity = torch.add(sumEgoVelocity, abs(egoVelocity))

    randomNum = random.uniform(0, 1)
    if randomNum > 0.5:
        counter = counter + 1
        skip = skip +1

    counter += 1
    
# length = lenth-skip
groundTruthNorm39 = torch.div(sumGroundTruth.view(60),length).to('cpu')
egoVelocityNorm39 = torch.div(sumEgoVelocity, length).to('cpu')
socialInformationOfCarsInViewNorm39 = torch.div(sumSocialInformationOfCarsInView.view(80), length).to('cpu')
encodedMapNorm39 = torch.div(sumEncidedMap.view(64), length).to('cpu')

print(f'egoVelocityNorm39 = {egoVelocityNorm39}')
print(f'groundTruthNorm39 = {groundTruthNorm39}')
print(f'socialInformationOfCarsInViewNorm39 = {socialInformationOfCarsInViewNorm39}')
print(f'encodedMapNorm39 = {encodedMapNorm39}')

torch.save(groundTruthNorm39, 'eviRelGroundTruthNorm39.pt')
torch.save(egoVelocityNorm39, 'eviEgoVelocityNorm39.pt')
torch.save(socialInformationOfCarsInViewNorm39, 'eviSocialInformationOfCarsInViewNorm39.pt')
torch.save(encodedMapNorm39, 'eviEncodedMapNorm39.pt')
