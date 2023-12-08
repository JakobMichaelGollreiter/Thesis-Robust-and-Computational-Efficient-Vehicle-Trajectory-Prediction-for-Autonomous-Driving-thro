
# for neuralNet
import torch
from torchinfo import summary
from torch.nn.functional import normalize
import numpy as np
import torch.nn as nn
import commonroad_prediction.toPointListConversion as toPointListConversion
import os


####################
wholeModelName = 'wholeNetPrototype5'
####################

class TrajectoryPredictionNet(nn.Module):
    def __init__(self):
        super(TrajectoryPredictionNet, self).__init__() 
        self.predictor1 = nn.Linear(145, 512)  # Future dx and dy predictor
        self.predictor2 = nn.Linear(512, 256)
        self.predictor3 = nn.Linear(256, 128)
        self.predictor4 = nn.Linear(128, 60)

    def forward(self, input_data):
        predicted_trajectory = self.predictor1(input_data)
        predicted_trajectory = nn.ReLU()(predicted_trajectory)
        predicted_trajectory = self.predictor2(predicted_trajectory)
        predicted_trajectory = nn.ReLU()(predicted_trajectory)
        predicted_trajectory = self.predictor3(predicted_trajectory)
        predicted_trajectory = nn.ReLU()(predicted_trajectory)
        predicted_trajectory = self.predictor4(predicted_trajectory)
        return predicted_trajectory.view(-1, 30 * 2)


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


class NeuralNets:
    def loadNeuralNets(self):
        """
        Load the pre-trained neural network models for trajectory prediction and autoencoder.

        Args:
        - None

        Returns:
        - autoencoder (ConvAutoencoder): The pre-trained autoencoder model.
        - model (TrajectoryPredictionNet): The pre-trained trajectory prediction model.
        """

        '''LOAD A TRAINED ENCODER'''
        autoencoder = ConvAutoencoder(encoder, decoder)
        # file_path = 'wholeNetModelsBuildUp/combined.csv_CNNModelFlatten32x1x1Combined_Epochs50_ValidationLoss:0.003411_lr:0.0005__2023_06_18_22:52:17.pth'
        # file_path = 'comparisonLinearVsCNNVsCNN/db39Model/notAugmented/combined.csv_CNNModelFlatten64x1x1CombinedTry_Epochs100_ValidationLoss:0.002493_lr:0.0003__2023_07_13_03:58:57.pth'
        file_path = 'comparisonLinearVsCNNVsCNN/db39Model/notAugmented/combined.csv_CNNModelFlatten64x1x1CombinedTry_Epochs200_ValidationLoss:0.001812_lr:0.0003__2023_07_19_18:02:20.pth' # This is the convolutional encoder model used in this thesis  
        autoencoder = torch.load(file_path, map_location=torch.device('cpu'))
        autoencoder.eval()

        '''LOAD A TRAINED MLP'''
        model = TrajectoryPredictionNet()
        # file_path = 'linearPredictionModels/db39/mega/combined.csv_wholeNetPrototype5_Epochs_3_ValidationLoss2.106801_lr0.0003_2023_08_04_09:39:10.pth'
        # file_path = 'linearPredictionModels/db39/mega/combined.csv_wholeNetPrototype5_Epochs_45_ValidationLoss4.482286_lr0.0003_2023_08_04_04:11:55.pth'
        # file_path = 'linearPredictionModels/db39/mega/combined.csv_wholeNetPrototype5_Epochs_10_ValidationLoss2.313236_lr0.0003_2023_08_04_13:15:32.pth'
        # file_path = 'linearPredictionModels/db39/mega/combined.csv_wholeNetPrototype5_Epochs_60_ValidationLoss2.230890_lr0.0003_2023_08_05_08:43:31.pth'
        # file_path = 'linearPredictionModels/cleanedCombine.csv_wholeNetPrototype5_Epochs_5_ValidationLoss0.476995_lr0.0006_2023_08_12_10:28:09.pth'
        # file_path = 'linearPredictionModels/db39/mega/cleanedCombine.csv_wholeNetPrototype5_Epochs_10_ValidationLoss0.418623_lr0.0006_2023_08_12_16:33:16.pth'
        # file_path = 'linearPredictionModels/db39/small/cleanedCombine.csv_wholeNetPrototype5_Epochs_10_ValidationLoss0.052175_lr0.0006_2023_08_12_20:50:53.pth'
        # file_path = 'linearPredictionModels/db39/mega/cleanedCombine.csv_wholeNetPrototype5_Epochs_25_ValidationLoss0.465854_lr0.0030_2023_08_13_05:44:45.pth'
        # file_path = 'linearPredictionModels/db39/small/cleanedCombine.csv_wholeNetPrototype5_Epochs_10_ValidationLoss0.052175_lr0.0006_2023_08_12_20:50:53.pth'
        # file_path = 'linearPredictionModels/db39/mega/cleanedCombine.csv_wholeNetPrototype5_Epochs_10_ValidationLoss0.378102_lr0.0003_2023_08_14_14:10:16.pth' 
        # file_path = 'linearPredictionModels/db39/mega/cleanedCombine.csv_wholeNetPrototype5_Epochs_25_ValidationLoss0.400800_lr0.0003_2023_08_15_08:05:43.pth' 
        # file_path = 'linearPredictionModels/db39/small/cleanedCombine.csv_wholeNetPrototype5_Epochs_25_ValidationLoss0.638217_lr0.0003_2023_08_15_09:30:48.pth' 
        # file_path = 'linearPredictionModels/db39/small/cleanedCombine.csv_wholeNetPrototype5_Epochs_25_ValidationLoss0.671118_lr0.0003_2023_08_15_09:58:31.pth' 
        # file_path = 'linearPredictionModels/db39/mega/cleanedCombine.csv_wholeNetPrototype5_Epochs_50_ValidationLoss0.417467_lr0.0003_2023_08_16_07:42:30.pth' 
        file_path = 'linearPredictionModels/db39/mega/cleanedCombine.csv_wholeNetPrototype5_Epochs_25_ValidationLoss0.442764_lr0.0003_2023_08_16_19:33:00.pth' # l1 + l2 loss (This is the MLP model used in the thesis)
        # file_path = 'linearPredictionModels/db39/mega/cleanedCombine.csv_wholeNetPrototype5_Epochs_25_ValidationLoss0.376590_lr0.0003_2023_08_17_06:15:32.pth' #only l1 loss
        # file_path = 'linearPredictionModels/db39/mega/cleanedCombine.csv_wholeNetPrototype5_Epochs_25_ValidationLoss0.368600_lr0.0003_2023_08_17_19:42:59.pth' # l1 + l2 loss ablation without social tensor

        model = torch.load(file_path, map_location=torch.device('cpu'))
        model.eval()
        summary(model, (1, 145))
        
        return autoencoder, model

