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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

groundTruthNorm = torch.load('Norms/eviRelGroundTruthNorm39.pt').to(device)
egoVelocityNorm = torch.load('Norms/eviEgoVelocityNorm39.pt').to(device)
socialInformationOfCarsInViewNorm = torch.load('Norms/eviSocialInformationOfCarsInViewNorm39.pt').to(device)
encodedMapNorm = torch.load('Norms/eviEncodedMapNorm39.pt').to(device)
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
        egoVelocity = torch.tensor(float(row['egoVelocity'])).to(device)
        time_begin = torch.tensor(int(row['time_begin'])).to(device)
        mapInView = torch.tensor(eval(row['mapInView'])).to(device)
        socialInformationOfCarsInView = torch.tensor(eval(row['socialInformationOfCarsInView'])).view(4*20).to(device)
        groundTruthTrajectory = torch.tensor(eval(row['groundTruthTrajectory'])).to(device)

        if (torch.any(torch.abs(torch.tensor(groundTruthTrajectory)) > 15) or torch.any(torch.abs(torch.tensor(socialInformationOfCarsInView)) > 50) or torch.any(torch.abs(torch.tensor(egoVelocity)) > 60)): # check for bad data and if so skip it
            return self.__getitem__(index + 1) # try next entry in the dataset (~1.3% corrupted data)__getitem__(index + 1) # try next entry in the dataset (~1.3% corrupted
        
        modGroundTruthTrajectory = torch.tensor([[entry[0], entry[1]] for entry in groundTruthTrajectory]).view(30,2).to(device) # ignore the last entry of the ground truth trajectory as this is just a number indicating how the trajectory was generated 

        ### Normalizing the input tensors
        modGroundTruthTrajectory = torch.where(modGroundTruthTrajectory != 0, modGroundTruthTrajectory / groundTruthNorm.view(30,2), torch.zeros_like(modGroundTruthTrajectory)) # normalizing the ground truth tensor pointList
        
        egoVelocity = torch.where(egoVelocity != 0, egoVelocity / egoVelocityNorm, torch.zeros_like(egoVelocity)) # normalizing the egoVelocity
        socialInformationOfCarsInView = torch.where(socialInformationOfCarsInView != 0, socialInformationOfCarsInView / socialInformationOfCarsInViewNorm, torch.zeros_like(socialInformationOfCarsInView)) # normalizing the socialInformationOfCarsInView

        sample = {
            'filename' : filename,
            'cr_id': cr_id,
            'time_begin': time_begin,
            'egoVelocity': torch.tensor(egoVelocity).to(device),
            'mapInView': torch.tensor(mapInView).to(device),
            'socialInformationOfCarsInView': torch.tensor(socialInformationOfCarsInView).to(device),
            'groundTruthTrajectory': torch.tensor(groundTruthTrajectory).to(device),
            'modGroundTruthTrajectory': torch.tensor(modGroundTruthTrajectory).to(device)
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
workingDirectory = current_directory

# dataBase = 'db39/small'
dataBase = 'db39/bigCC2smallTest'
# dataBase = 'db39/bigC'
# dataBase = 'db39/trainLocal'

datasetPath = os.path.join(dataBase, 'combined.csv')
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
datasetPath = os.path.join(dataBase, 'combined.csv')
datasetPath = os.path.join(workingDirectory, datasetPath)
valDataset = CustomDataset(datasetPath) # create dataset cobined data
num_data_items = len(valDataset)
print("Number of data items:", num_data_items)
print(valDataset.__getitem__(0)['modGroundTruthTrajectory'].shape)


def orientVectorList(vectorList):
    orientVectorList = np.zeros((30, 2))
    relativeAngle = np.arctan2(vectorList[0][1], vectorList[0][0]) 
    orientVectorList[0][0] = vectorList[0][0] 
    orientVectorList[0][1] = vectorList[0][1] 

    for i in range(len(vectorList)-1):
        dx = vectorList[i+1][0]
        dy = vectorList[i+1][1]
        if (dx <= 0.00001):
            angle = np.arctan2(0, 1) 
        else:
            angle = np.arctan2(dy, dx) # ToDo: check if this is correct
        relativeAngle = angle + relativeAngle
        


        orientVectorList[i+1][0] = dx * np.cos(relativeAngle) - dy * np.sin(relativeAngle)
        orientVectorList[i+1][1] = dx * np.sin(relativeAngle) + dy * np.cos(relativeAngle)

    return orientVectorList
    
def convertingToAbsolutePointList(orientVectorList):
    x_last = 0.0
    y_last = 0.0
    x = x_last
    y = y_last
    pointList = np.zeros((30, 2))

    # Plot each x and y pair on the axis
    for i in range(len(orientVectorList)):
        x_last, y_last = x, y
        x, y = x_last + orientVectorList[i][0], y_last + orientVectorList[i][1]
        pointList[i][0] = x
        pointList[i][1] = y
    return pointList

def convertRelativeDxDyTensorAbsolutePointList(relativeDxDyList):
    relativeDxDyWithOrientetAnglesList = orientVectorList(relativeDxDyList.view(30, 2).tolist())
    orientedPointList = convertingToAbsolutePointList(relativeDxDyWithOrientetAnglesList)
    return orientedPointList

def makeInputTensorValidation(encodedMap, egoVelocity, socialInformationOfCarsInView):
        egoVelocity = torch.tensor(egoVelocity)  # Convert egoVelocity to a tensor
        combined_tensor = torch.cat((encodedMap.view(-1,64), egoVelocity.view(-1,1), socialInformationOfCarsInView.view(-1,4*20)), dim=1) # Concatenate flattened tensors
        # print(combined_tensor.shape)
        inputTensor = combined_tensor
        return inputTensor
    
def outputTrajectoryOnDrivableRoadScalar(outputPointList, mapInView):
    metaFactor = (input_sizeX)/27
    mapInView = np.flipud(mapInView.cpu())
    unnormedOutputPointList = torch.where(outputPointList != 0, outputPointList * groundTruthNorm.view(60), torch.zeros_like(outputPointList)) # unnormalizing the ground truth tensor pointList

    outputPointList = convertRelativeDxDyTensorAbsolutePointList(unnormedOutputPointList)
    rotation_matrix = np.array([[0, -1], [1, 0]]) # Rotate the vectors counterclockwise by 90 degrees
    outputPointList = np.dot(outputPointList, rotation_matrix.T)
    overlapScalar = 0
    outsideBoundary = 0
    for i in range(len(outputPointList)):
        if np.isnan(outputPointList[i][0]).any() or np.isnan(outputPointList[i][1]).any(): 
            if(i-outsideBoundary == 0): return 0.01
            return overlapScalar/(i-outsideBoundary)

        matrixY = int(round(outputPointList[i][0]*metaFactor) + int((input_sizeX/2) -1))
        matrixX = int(round(outputPointList[i][1]*metaFactor))

        if(matrixX >= 0 and matrixX < input_sizeX-1 and matrixY >= 0 and matrixY < input_sizeX-1):
            if(matrixX == 0 or matrixX == input_sizeX-1 or matrixY == 0 or matrixY == input_sizeX-1): 
                if(mapInView[matrixX][matrixY] == 1):
                    overlapScalar += 1
                else:   
                    overlapScalar += 0
            else:
                activation = 0 # This is a 3x3 kernel with activation in the middle and the corners [0.1 x 0 x 0.1] | [0 x 4.6 x 0] | [0.1 x 0 x 0.1]
                if(mapInView[matrixX][matrixY] == 1):
                    activation = activation + 4.0       
                if(mapInView[matrixX-1][matrixY-1] == 1):
                    activation = activation + 0.25        
                if(mapInView[matrixX-1][matrixY+1] == 1):
                    activation = activation + 0.25       
                if(mapInView[matrixX+1][matrixY-1] == 1):
                    activation = activation + 0.25        
                if(mapInView[matrixX+1][matrixY+1] == 1):
                    activation = activation + 0.25   
                overlapScalar += activation/5    
        else:
            outsideBoundary += 1 # outside the 27x27m boundary  
    lengthInsideBoundary = (len(outputPointList)-outsideBoundary)    
    if(lengthInsideBoundary != 0):
        return overlapScalar/lengthInsideBoundary
    return overlapScalar/0.01
    # train model linear
batch_size = 32  # Set your desired batch size
test_batch_size = 1 # ToDo test batch size. Here should be no batch

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Create data loaders for the training set
validation_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Create data loaders for the training set
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True) # Create data loaders for the test set

# 733 interesting crossing
sampleSimple = valDataset.__getitem__(55)
sampleEasy = valDataset.__getitem__(65)
sampleMedium = valDataset.__getitem__(50)
sampleHard = valDataset.__getitem__(35)
validationImages = [sampleSimple, sampleEasy, sampleMedium, sampleHard]
for image in validationImages:
    image['modGroundTruthTrajectory']
validationImagesLosses = [0.0, 0.0, 0.0, 0.0]


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
file_path = os.path.join(workingDirectory,'db39/imageTrain2/combined.csv_CNNModelFlatten64x1x1CombinedTry_Epochs200_ValidationLoss:0.001812_lr:0.0003__2023_07_19_18:02:20.pth')

learntCNN = ConvAutoencoder(encoder, decoder).to(device)
learntCNN = torch.load(file_path, map_location=torch.device('cuda'))
learntCNN.eval()


####################
wholeModelName = 'wholeNetPrototype5'
####################


# Define the network architecture
class TrajectoryPredictionNet(nn.Module):
    def __init__(self):
        super(TrajectoryPredictionNet, self).__init__() 
        self.predictor1 = nn.Linear(145, 512)  # Future dx and dy predictor
        self.predictor2 = nn.Linear(512, 60)

    def forward(self, input_data):
        predicted_trajectory = self.predictor1(input_data)
        predicted_trajectory = nn.ReLU()(predicted_trajectory)
        predicted_trajectory = self.predictor2(predicted_trajectory)
        # predicted_trajectory = nn.ReLU()(predicted_trajectory) # no relu at the output !
        return predicted_trajectory.view(-1, 30 * 2)


model = TrajectoryPredictionNet().to(device)
    

# specify loss function
# criterion = torch.nn.MSELoss()
criterion = torch.nn.L1Loss()
# specify loss function
lr = 3e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)    

model.train()
epochs = 50  # Set your desired number of epochs
print(f'Training {wholeModelName} for {epochs} epochs:')
train_loss_history = []
validation_loss_history = []
potence = 2
for epoch in range(epochs):
    train_loss = 0.0
    valid_loss = 0.0
    for batch in train_loader: # training loop
        mapInView = batch['mapInView'].to(device)
        images = batch['mapInView'].view(-1, input_sizeX , input_sizeY).to(device)
        images = images.unsqueeze(1)
        encodedMap = learntCNN.encoder.forward(images).to(device) # Output of encoder dimensionality reduction
        # print(f'encoded map is on GPU{encodedMap.is_cuda}')
        # print(f'encoded map Norm is on GPU{encodedMapNorm.is_cuda}')
        encodedMap = torch.where(encodedMap != 0, encodedMap / encodedMapNorm, torch.zeros_like(encodedMap)).view(-1,64).to(device) # normalizing the encodedMap
        egoVelocity = batch['egoVelocity']
        socialInformationOfCarsInView = batch['socialInformationOfCarsInView']
        modGroundTruthTrajectory = batch['modGroundTruthTrajectory']
        inputTensor = makeInputTensorValidation(encodedMap, egoVelocity, socialInformationOfCarsInView).view(-1,145).to(device)
        output = model.forward(inputTensor)  # Forward pass
        onRoad = 0
        for i in range(len(output)):
            single_output = output[i]
            val = outputTrajectoryOnDrivableRoadScalar(single_output, mapInView[i])
            # if val == 0.1:
                # print(f'scenario {i} = {batch["filename"][i]} , time = {batch["time_begin"][i]}, car = {batch["cr_id"][i]}')
            onRoad = onRoad + val
        onRoad = onRoad / len(output)


        groundTruthTrajectory = batch['groundTruthTrajectory']
        length = 30
        lossOutput = output.view(30, -1, 2)[:length]
        lossModGroundTruthTrajectory = modGroundTruthTrajectory.view(30, -1, 2)[:length]

        loss = criterion(lossOutput.view(-1,length,2), lossModGroundTruthTrajectory.view(-1,length,2))/(onRoad**potence)

        optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update the parameters
        train_loss += loss.item()

    train_loss = train_loss / len(train_dataset) 
    train_loss_history.append(train_loss) 
    
    with torch.no_grad():
        i = 0
        for sample in validationImages: # this loop is just for the visualisation of the same 4 validation images to detect progress over the epochs
            images = sample['mapInView'].view(-1, input_sizeX , input_sizeY).to(device)
            images = images.unsqueeze(1)
            encodedMap = learntCNN.encoder.forward(images).to(device) # Output of encoder dimensionality reduction
            # print(f'encoded map is on GPU{encodedMap.is_cuda}')
            # print(f'encoded map Norm is on GPU{encodedMapNorm.is_cuda}')
            encodedMap = torch.where(encodedMap != 0, encodedMap / encodedMapNorm, torch.zeros_like(encodedMap)).view(-1,64).to(device) # normalizing the encodedMap
            egoVelocity = sample['egoVelocity']
            socialInformationOfCarsInView = sample['socialInformationOfCarsInView']
            modGroundTruthTrajectory = sample['modGroundTruthTrajectory']
            inputTensor = makeInputTensorValidation(encodedMap, egoVelocity, socialInformationOfCarsInView).view(-1,145).to(device)
            output = model.forward(inputTensor)  # Forward pass
            onRoad = outputTrajectoryOnDrivableRoadScalar(output.view(30*2), sample['mapInView'].view(input_sizeX , input_sizeY))
            loss = criterion(output.view(30,2), modGroundTruthTrajectory.view(30,2))/(onRoad**potence) 

            output = output.view(30*2)
            modGroundTruthTrajectory = modGroundTruthTrajectory.view(30*2)
            sample['output'] = output
            sample['modGroundTruthTrajectory'] = modGroundTruthTrajectory
            validationImagesLosses[i] = loss.item()
            i += 1
        valid_loss = sum(validationImagesLosses) / len(validationImages) 
        validation_loss_history.append(valid_loss) 


    print('   Epoch: [{:2d}|{:2d}]   Train Loss: {:.7f}  Validation Loss: {:.7f}'.format(epoch + 1, epochs, train_loss, valid_loss))

print('DONE TRAINING')


# save model

# Create the parent directory if it doesn't exist
parent_dir = 'wholeNetModels'
os.makedirs(parent_dir, exist_ok=True)

now = datetime.datetime.now()
timestamp = now.strftime("%Y_%m_%d_%H:%M:%S")
# Save the model
file_name = f'{os.path.dirname(datasetPath)}_{wholeModelName}_Epochs{epochs:}_ValidationLoss:{float(validation_loss_history[-1]):.6f}_lr:{lr:.4f}__{timestamp}.pth'
file_path = os.path.join(parent_dir, file_name)
torch.save(model, file_path)
print(f'Model saved to {file_path}')
