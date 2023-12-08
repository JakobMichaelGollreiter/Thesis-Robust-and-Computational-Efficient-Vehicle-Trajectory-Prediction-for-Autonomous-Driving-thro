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
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import datetime
import os
from tqdm import tqdm
import random


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


groundTruthNorm = torch.load('NormsTest/eviRelGroundTruthNorm39.pt').to(device)
egoVelocityNorm = torch.load('NormsTest/eviEgoVelocityNorm39.pt').to(device)
socialInformationOfCarsInViewNorm = torch.load('NormsTest/eviSocialInformationOfCarsInViewNorm39.pt').to(device)
encodedMapNorm = torch.load('NormsTest/eviEncodedMapNorm39.pt').to(device)
# encodedMapNorm.requires_grad = False
# Overwrite all the entries with 1.0
encodedMapNorm.fill_(1.0).to(device)
print(f'groundTruthNorm = {groundTruthNorm.view(30,2)}')
print(f'eogVelocityNorm = {egoVelocityNorm}')
print(f'socialInformationOfCarsInViewNorm = {socialInformationOfCarsInViewNorm}')
print(f'encodedMapNorm = {encodedMapNorm}')

class CustomDataset(Dataset): # create Dataset class
    def __init__(self, csv_file):
        self.data = self.load_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        filename = str(row['\ufefffilename'])
        cr_id = int(row['cr_id'])
        egoVelocity = torch.tensor(float(row['egoVelocity']))
        time_begin = int(row['time_begin'])
        mapInView = eval(row['mapInView'])
        socialInformationOfCarsInView = torch.tensor(eval(row['socialInformationOfCarsInView'])).view(4*20).to(device)
        groundTruthTrajectory = eval(row['groundTruthTrajectory'])

        modGroundTruthTrajectory = torch.tensor([[entry[0], entry[1]] for entry in groundTruthTrajectory]).view(30,2).to(device) # ignore the last entry of the ground truth trajectory as this is just a number indicating how the trajectory was generated 
 
        ### Normalizing the input tensors
        modGroundTruthTrajectory = torch.where(modGroundTruthTrajectory != 0, modGroundTruthTrajectory / groundTruthNorm.view(30,2), torch.zeros_like(modGroundTruthTrajectory)).to(device) # normalizing the ground truth tensor pointList
        egoVelocity = torch.where(egoVelocity != 0, egoVelocity / egoVelocityNorm, torch.zeros_like(egoVelocity)).to(device) # normalizing the egoVelocity
        socialInformationOfCarsInView = torch.where(socialInformationOfCarsInView != 0, socialInformationOfCarsInView / socialInformationOfCarsInViewNorm, torch.zeros_like(socialInformationOfCarsInView)) # normalizing the socialInformationOfCarsInView
        

        sample = {
            'filename' : filename,
            'cr_id': cr_id,
            'time_begin': time_begin,
            'egoVelocity': egoVelocity,
            'mapInView': torch.tensor(mapInView),
            'socialInformationOfCarsInView': socialInformationOfCarsInView,
            'groundTruthTrajectory': torch.tensor(groundTruthTrajectory),
            'modGroundTruthTrajectory': modGroundTruthTrajectory
            
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
workingDirectory = current_directory

# dataBase = 'db39/bigCsmall3'
# dataBase = 'db39/small'
# dataBase = 'db39/trainLocal'
dataBase = 'db39/mega'
# dataBase = 'db39/trainPredictionLocal'
# dataBase = 'db39/bigC'
# dataBase = 'db39/newHorizon'
datasetPath = os.path.join(dataBase, 'cleanedCombine.csv')
# datasetPath = os.path.join(dataBase, 'combined.csv')
datasetPath = os.path.join(workingDirectory, datasetPath)
dataset = CustomDataset(datasetPath) # create dataset cobined data
sample = dataset.__getitem__(0)
input_sizeX = sample['mapInView'].shape[0] 
input_sizeY = sample['mapInView'].shape[1]

# Define the proportions for train, validation, and test sets (e.g., 70%, 15%, 15%)
train_ratio = 0.98
val_ratio = 0.01
test_ratio = 0.01
# Calculate the number of samples for each set
num_samples = len(dataset)
train_size = int(train_ratio * num_samples)
val_size = int(val_ratio * num_samples)
test_size = num_samples - train_size - val_size
# Use random_split to split the dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
num_data_items = len(dataset)
print("Number of data items:", num_data_items)


current_directory = os.getcwd()  # Get the current working directory
workingDirectory = current_directory
dataBase = 'db39/validation'
datasetPath = os.path.join(dataBase, 'cleanedCombine.csv')
datasetPath = os.path.join(workingDirectory, datasetPath)
validation_dataset = CustomDataset(datasetPath) # create dataset cobined data
num_data_items = len(validation_dataset) # create dataset cobined data

print("Number of data items:", num_data_items)
print(validation_dataset.__getitem__(0)['modGroundTruthTrajectory'].shape)


sampleSimple = validation_dataset.__getitem__(55)
sampleEasy = validation_dataset.__getitem__(65)
sampleMedium = validation_dataset.__getitem__(50)
sampleHard = validation_dataset.__getitem__(35)
validationImages = [sampleSimple, sampleEasy, sampleMedium, sampleHard]

# def orientVectorListTorchOld(vectorList):
#     # orientVectorList = torch.zeros((30, 2))
#     orientVectorList = torch.zeros((30, 2)).to(device)
#     relativeAngle = torch.atan2(vectorList[0][1], vectorList[0][0]).to(device)
#     orientVectorList[0][0] = vectorList[0][0]
#     orientVectorList[0][1] = vectorList[0][1]

#     for i in range(len(vectorList)-1):
#         dx = vectorList[i+1][0]
#         dy = vectorList[i+1][1]
#         if torch.abs(dx) < 1e-8 and torch.abs(dy) < 1e-8:
#             angle = torch.tensor(0.0)
#         else:
#             angle = torch.atan2(dy, dx)
#         relativeAngle = angle + relativeAngle

#         orientVectorList[i+1][0] = dx * torch.cos(relativeAngle) - dy * torch.sin(relativeAngle)
#         orientVectorList[i+1][1] = dx * torch.sin(relativeAngle) + dy * torch.cos(relativeAngle)

#     return orientVectorList

def orientVectorListTorch(vectorList):
    dx, dy = vectorList[:,0], vectorList[:,1]
    angle_increment = torch.atan2(dy, dx).to(device)
    
    mask = (torch.abs(dx) < 1e-8) & (torch.abs(dy) < 1e-8)
    angle_increment[mask] = 0
    relativeAngle = torch.cumsum(angle_increment, dim=0) 
    relativeAngle[0] = 0
    orientVectorList = torch.stack([
        dx * torch.cos(relativeAngle) - dy * torch.sin(relativeAngle),
        dx * torch.sin(relativeAngle) + dy * torch.cos(relativeAngle)
    ], dim=1)
    return orientVectorList

# def convertingToAbsolutePointListTorchOld(orientVectorList):
#     x_last = torch.tensor(0.0)
#     y_last = torch.tensor(0.0)
#     x = x_last
#     y = y_last
#     # pointList = torch.zeros((30, 2))
#     pointList = torch.zeros((30, 2)).to(device)

#     for i in range(len(orientVectorList)):
#         x_last, y_last = x, y
#         x, y = x_last + orientVectorList[i][0], y_last + orientVectorList[i][1]
#         pointList[i][0] = x
#         pointList[i][1] = y
#     return pointList

def convertingToAbsolutePointListTorch(orientVectorList):
    return torch.cumsum(orientVectorList, dim=0)

def convertRelativeDxDyTensorAbsolutePointListTorch(relativeDxDyList):
    # relativeDxDyWithOrientetAnglesList = orientVectorListTorchOld(relativeDxDyList.view(30, 2))
    relativeDxDyWithOrientetAnglesList = orientVectorListTorch(relativeDxDyList.view(30, 2))
    # print(relativeDxDyWithOrientetAnglesList)
    # print(relativeDxDyWithOrientetAnglesList2)

    # orientedPointList = convertingToAbsolutePointListTorchOld(relativeDxDyWithOrientetAnglesList)
    orientedPointList = convertingToAbsolutePointListTorch(relativeDxDyWithOrientetAnglesList)

    return orientedPointList

# def lengthInsideMapInViewOld(trajectory):
#     boundaryXMin = torch.tensor(0)
#     boundaryXMax = torch.tensor(27)
#     boundaryYMin = torch.tensor(-13.5)  # Use a dot (.) for the decimal point
#     boundaryYMax = torch.tensor(13.5)   # Use a dot (.) for the decimal point

#     trajectory = trajectory.view(30, 2) # reshaping the tensor to a list of points
#     for i in range(len(trajectory)):
#         if trajectory[i][0] < boundaryXMin or trajectory[i][0] > boundaryXMax or trajectory[i][1] < boundaryYMin or trajectory[i][1] > boundaryYMax:
#             return i+1
        
#     return 30

def lengthInsideMapInView(trajectory):
    x,y = trajectory[:,0], trajectory[:,1]
    mask = (x < 0) | (x > 27) | (y < -13.5) | (y > 13.5)
    idx = torch.nonzero(mask)
    return 30 if idx.numel() == 0 else idx[0].item() + 1

def chamfLoss(trajectoyPoints, mapPoints):
    # trajectoyPoints (N, 2)
    # mapPoints (M, 2)
    
    assert torch.is_tensor(trajectoyPoints) and torch.is_tensor(mapPoints)
    dists = (trajectoyPoints[:, None] - mapPoints[None])**2  # (N, M, 2)
    dists = torch.sqrt(torch.sum(dists, axis=-1))  # (N, M)
    dists = torch.min(dists, axis=1).values  # (N,)
    return dists


def trajectoryOnRoad(trajectory, mapInView):
    viewLength = 27
    arraySubcornerLength = viewLength / float(39-1)
    mapInView = torch.flip(mapInView, [0]).to(device)  # Use PyTorch flip
    unNormedTrajectory = torch.where(trajectory != 0, trajectory * groundTruthNorm.view(60), torch.zeros_like(trajectory)) # unnormalizing the ground truth tensor pointList
    trajectory = convertRelativeDxDyTensorAbsolutePointListTorch(unNormedTrajectory)
    lengthInsideMap = lengthInsideMapInView(trajectory)
    roadPoints = torch.where(mapInView == 1)

    x,y = roadPoints[0], roadPoints[1]  
    points = torch.stack([x * arraySubcornerLength,viewLength/2 - y * arraySubcornerLength], axis=1)

    points = points.to(device)
    trajectory = trajectory.view(30,2)[:lengthInsideMap]
    dists = chamfLoss(trajectory, points)
    # print(dists.grad_fn)
    dists = torch.where(dists < 0.55, dists * 0.000001, dists)
    dist = torch.sum(dists)/lengthInsideMap
    return dist, lengthInsideMap

# train model linear
batch_size = 128  # Set your desired batch size
test_batch_size = 1 # ToDo test batch size. Here should be no batch

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True) # Create data loaders for the training set
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, drop_last=True) # Create data loaders for the training set
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True) # Create data loaders for the test set

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
    
encoder = Encoder().to(device)
decoder = Decoder().to(device)
# file_path = os.path.join(workingDirectory,'comparisonLinearVsCNNVsCNN/db39Model/notAugmented/combined.csv_CNNModelFlatten64x1x1CombinedTry_Epochs100_ValidationLoss:0.002493_lr:0.0003__2023_07_13_03:58:57.pth')
file_path = os.path.join(workingDirectory,'db39/imageTrain2/combined.csv_CNNModelFlatten64x1x1CombinedTry_Epochs200_ValidationLoss:0.001812_lr:0.0003__2023_07_19_18:02:20.pth')
# file_path = os.path.join(workingDirectory,'../comparisonLinearVsCNNVsCNN/db39Model/notAugmented/combined.csv_CNNModelFlatten64x1x1CombinedTry_Epochs200_ValidationLoss:0.001812_lr:0.0003__2023_07_19_18:02:20.pth')
learntCNN = ConvAutoencoder(encoder, decoder).to(device)
learntCNN = torch.load(file_path, map_location=torch.device('cuda'))
# learntCNN = torch.load(file_path, map_location=torch.device('cpu'))
learntCNN.eval()

####################
wholeModelName = 'wholeNetPrototype5'
####################

# Define the network architecture
class TrajectoryPredictionNet(nn.Module):
    def __init__(self):
        super(TrajectoryPredictionNet, self).__init__() 
        self.predictor1 = nn.Linear(65, 512)  # Future dx and dy predictor
        self.predictor2 = nn.Linear(512, 256)
        self.predictor3 = nn.Linear(256, 128)
        self.predictor4 = nn.Linear(128, 60)
        self.relu = nn.ReLU()


    def forward(self, input_data):
        predicted_trajectory = self.predictor1(input_data)
        predicted_trajectory = self.relu(predicted_trajectory)
        predicted_trajectory = self.predictor2(predicted_trajectory)
        predicted_trajectory = self.relu(predicted_trajectory)
        predicted_trajectory = self.predictor3(predicted_trajectory)
        predicted_trajectory = self.relu(predicted_trajectory)
        predicted_trajectory = self.predictor4(predicted_trajectory)
        return predicted_trajectory.view(-1, 30 * 2)
    
model = TrajectoryPredictionNet().to(device)
# file_path = os.path.join(workingDirectory,'db39/cleanedCombine.csv_wholeNetPrototype5_Epochs_4_ValidationLoss0.465718_lr0.0006_2023_08_11_21/59/56.pth')
# model = torch.load(file_path, map_location=torch.device('cuda'))
model.eval()
summary(model, (1, 65))


# specify loss function
#criterion = torch.nn.MSELoss()
criterion = torch.nn.L1Loss()
# specify loss function
lr = 3e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def makeInputTensorValidation(encodedMap, egoVelocity, socialInformationOfCarsInView):
    egoVelocity = egoVelocity  # Convert egoVelocity to a tensor
    combined_tensor = torch.cat((encodedMap.view(-1,64), egoVelocity.view(-1,1)), dim=1) # Concatenate flattened tensors
    inputTensor = combined_tensor
    return inputTensor
    
model.train()
epochs = 25  # Set your desired number of epochs
print(f'Training {wholeModelName} for {epochs} epochs:')
train_loss_history = []
validation_loss_history = []
validation_images_loss_history = []


for epoch in tqdm(range(epochs), desc='Training Epoch', unit='epoch'):
    train_loss = 0.0
    valid_loss = 0.0
    validation_test_images_loss = 0.0

    for batch in tqdm(train_loader, desc='Training…', unit='batch', leave=False):
        mapInView = batch['mapInView'].to(device)
        images = batch['mapInView'].view(-1, input_sizeX , input_sizeY).to(device) 
        images = images.unsqueeze(1)
        encodedMap = learntCNN.encoder.forward(images).to(device) # Output of encoder dimensionality reduction
        egoVelocity = batch['egoVelocity'].to(device)
        socialInformationOfCarsInView = batch['socialInformationOfCarsInView'].to(device)
        modGroundTruthTrajectory = batch['modGroundTruthTrajectory'].to(device)
        inputTensor = makeInputTensorValidation(encodedMap, egoVelocity, socialInformationOfCarsInView).view(-1,145)
        output = model(inputTensor)  # Forward pass
        output = output.view(-1,30,2)

        absX = torch.abs(output[:, :, 0])  # Shape: (4, 30)
        y = output[:, :, 1]
        output = output = torch.stack((absX, y), dim=2).view(-1,30,2)  # Combining the results back into a tensor with shape (30, 2)
        # output.requires_grad_()

        onRoadTensor = torch.zeros(len(output))  # Initialize a tensor to store onRoad values
        roadTensor = torch.zeros(len(output))  # Initialize a tensor to store road values

        for i in range(len(output)):
            onRoad, lengthInsideMap = trajectoryOnRoad(output[i].view(30 * 2) , mapInView[i].view(input_sizeX , input_sizeY))
            road = torch.sum(torch.full((lengthInsideMap,), 0, dtype=torch.float))/lengthInsideMap
            onRoadTensor[i] = onRoad
            roadTensor[i] = road

        loss1 = criterion(output[:lengthInsideMap], modGroundTruthTrajectory[:lengthInsideMap])
        loss2 = criterion(onRoadTensor, roadTensor)
        loss = loss1 + loss2*0.2
        if epoch > 0:
            optimizer.zero_grad()  # Zero the gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update the parameters
        train_loss += loss.item()


    train_loss = train_loss / (len(train_dataset)/batch_size) 
    train_loss_history.append(train_loss) 
    
    with torch.no_grad():
        for batch in tqdm(validation_loader, desc='Validation…', unit='batch', leave=False):  # this loop is just for the visualisation of the same 4 validation images to detect progress over the epochs
            mapInView = batch['mapInView'].to(device)
            images = batch['mapInView'].view(-1, input_sizeX , input_sizeY).to(device) 
            images = images.unsqueeze(1)
            encodedMap = learntCNN.encoder(images).to(device) # Output of encoder dimensionality reduction
            egoVelocity = batch['egoVelocity'].to(device)
            socialInformationOfCarsInView = batch['socialInformationOfCarsInView'].to(device)
            modGroundTruthTrajectory = batch['modGroundTruthTrajectory'].to(device)
            inputTensor = makeInputTensorValidation(encodedMap, egoVelocity, socialInformationOfCarsInView).view(-1,145).to(device)
            output = model(inputTensor).to(device)  # Forward pass
            output = output.view(-1,30,2)

            absX = torch.abs(output[:, :, 0])   # Shape: (4, 30)
            y = output[:, :, 1]
            output = output = torch.stack((absX, y), dim=2).view(-1,30,2)  # Combining the results back into a tensor with shape (30, 2)

            # Initialize lists to store tensors
            onRoadTensor = torch.zeros(len(output)).to(device)  # Initialize a tensor to store onRoad values
            roadTensor = torch.zeros(len(output)).to(device)  # Initialize a tensor to store road values

            for i in range(len(output)):
                onRoad, lengthInsideMap = trajectoryOnRoad(output[i].view(30 * 2) , mapInView[i].view(input_sizeX , input_sizeY))
                road = torch.sum(torch.full((lengthInsideMap,), 0, dtype=torch.float))/lengthInsideMap
                # print(onRoad)
                onRoadTensor[i] = onRoad
                roadTensor[i] = road

            loss1 = criterion(output[:lengthInsideMap], modGroundTruthTrajectory[:lengthInsideMap])
            loss2 = criterion(onRoadTensor, roadTensor)
            loss = loss1 + loss2*0.2
            valid_loss += loss.item()

        valid_loss = valid_loss / (len(validation_dataset)/batch_size) 
        validation_loss_history.append(valid_loss) 

        i = 0
        for sample in validationImages: # this loop is just for the visualisation of the same 4 validation images to detect progress over the epochs
            images = sample['mapInView'].view(-1, input_sizeX , input_sizeY) 
            images = images.unsqueeze(1).to(device)
            encodedMap = learntCNN.encoder(images).to(device) # Output of encoder dimensionality reduction
            egoVelocity = sample['egoVelocity'].to(device)
            socialInformationOfCarsInView = sample['socialInformationOfCarsInView'].to(device)
            modGroundTruthTrajectory = sample['modGroundTruthTrajectory'].to(device)
            inputTensor = makeInputTensorValidation(encodedMap, egoVelocity, socialInformationOfCarsInView).view(-1,145).to(device)
            output = model(inputTensor).to(device)  # Forward pass

            ###
            output = output.view(30,2) # Separating the two columns
            columnx = output[:, 0]
            columny = output[:, 1]
            
            abs_columnx = torch.abs(columnx) # Computing the absolute values for x column (only positive driving direction allowed) -> more stable training
            output = torch.stack((abs_columnx, columny), dim=1).view(30,2) # Combining the results back into a tensor with shape (30, 2)
            output = output.view(30*2)
            ###

            onRoad, lengthInsideMap = trajectoryOnRoad(output.view(60),sample['mapInView'].view(input_sizeX , input_sizeY))
            # onRoad = onRoad*egoVelocity
            road = torch.sum(torch.full((lengthInsideMap,), 0, dtype=torch.float))/lengthInsideMap
            output = output.view(30,2)
            modGroundTruthTrajectory = modGroundTruthTrajectory.view(30,2)

            loss1 = criterion(output[:lengthInsideMap], modGroundTruthTrajectory[:lengthInsideMap])
            loss2 = criterion(onRoad, road)
            loss = loss1 + loss2*0.2
      
            validation_test_images_loss += loss.item()
            i += 1
        validation_test_images_loss = validation_test_images_loss / len(validationImages) 
        validation_images_loss_history.append(validation_test_images_loss)     
        
    print('   Epoch: [{:2d}|{:2d}]   Train Loss: {:.7f}  Validation Loss: {:.7f}  Validation Test Images: {:.7f}'.format(epoch + 1, epochs, train_loss, valid_loss, validation_test_images_loss))
    
print('DONE TRAINING')

# Plot the loss history
plt.plot(train_loss_history, label='Training loss')
plt.plot(validation_loss_history, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()


#Update the parent directory to 'db39'
parent_dir = '/home/ubuntu/jakob_ba/predictionModelServer/db39'
os.makedirs(parent_dir, exist_ok=True)

# Save the model with the combined new name
now = datetime.datetime.now()
timestamp = now.strftime("%Y_%m_%d_%H:%M:%S")

# Modify the file name construction
file_name = f'{os.path.basename(datasetPath)}_{wholeModelName}_Epochs_{epochs}_ValidationLoss{float(validation_loss_history[-1]):.6f}_lr{lr:.4f}_{timestamp}.pth'
file_path = os.path.join(parent_dir, file_name)

# Save the model
torch.save(model, file_path)
print(f'Model saved to {file_path}')

# Save the loss plot
loss_plot_path = os.path.join(parent_dir, 'loss_plot.png')
plt.savefig(loss_plot_path)
print(f'train validation loss plot saved to {loss_plot_path}')

