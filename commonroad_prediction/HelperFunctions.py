
from configparser import ConfigParser
import math
import numpy as np
import torch

''' configuration from config.ini file: '''
file = 'commonroad_prediction/config.ini'

config = ConfigParser()
config.read(file)

arrayShape = eval(config['var']['arrayShape'])
oneDimension = eval(config['var']['oneDimension'])
viewLength = eval(config['var']['viewLength'])
arraySubcornerLength = eval(config['var']['arraySubcornerLength'])
toleranceOfDetection = eval(config['var']['toleranceOfDetection'])
masterTimesteps = eval(config['var']['masterTimesteps'])
velocityArrowScale = eval(config['var']['velocityArrowScale'])
cutOffTimeSteps = eval(config['var']['cutOffTimeSteps'])
buildImagesAndVideo = eval(config['var']['buildImagesAndVideo'])
egoVehicleCrID = eval(config['var']['egoVehicleCrID'])


def angle_difference(a1, a2):
    # Compute the anglåe between two angles, in radians
    return min((2 * math.pi) - abs(a1 - a2), abs(a1 - a2))


def normalize_value(value, min_value, max_value):
    normalized_value = (value - min_value) / (max_value - min_value)
    return normalized_value

def distance(element):
    x = element[0]
    y = element[1]
    if(x == 0 and y == 0):
        return 100000
    return math.sqrt(x**2 + y**2)

def calculate_relativePositionToEgo(absolutePositionToEgo, orientationOfEgo):
    orientationOfEgo = -orientationOfEgo
    dxPrime = absolutePositionToEgo[0] * math.cos(
        orientationOfEgo) - absolutePositionToEgo[1] * math.sin(orientationOfEgo)
    dyPrime = absolutePositionToEgo[0] * math.sin(
        orientationOfEgo) + absolutePositionToEgo[1] * math.cos(orientationOfEgo)
    return [dxPrime, dyPrime]


def calculate_relativeDxDyVelocityToEgo(velocity, relativeAngleToEgo):
    dxPrimeVelocity = velocity * math.cos(relativeAngleToEgo)
    dyPrimVelocity = velocity * math.sin(relativeAngleToEgo)
    return [dxPrimeVelocity, dyPrimVelocity]


def is_current_position_a_hit(x, y, statesOfVehicles, car):
    return (
        statesOfVehicles[car][0][0] >= x - toleranceOfDetection
        and statesOfVehicles[car][0][0] <= x + toleranceOfDetection
        and statesOfVehicles[car][0][1] >= y - toleranceOfDetection
        and statesOfVehicles[car][0][1] <= y + toleranceOfDetection
    )


def ground_truth_is_current_position_a_hit(x, y, xg, yg):
    return (
        xg >= x - toleranceOfDetection
        and xg <= x + toleranceOfDetection
        and yg >= y - toleranceOfDetection
        and yg <= y + toleranceOfDetection
    )


def calculate_position_in_oriented_coordinate_system(orientation, location, ix, iy):

    y = location[1] + math.sin(orientation) * (- viewLength*0.5 + iy*arraySubcornerLength) + \
        math.cos(orientation) * (viewLength - ix*arraySubcornerLength)
    x = location[0] + math.cos(orientation) * (- viewLength*0.5 + iy*arraySubcornerLength) - \
        math.sin(orientation) * (viewLength - ix*arraySubcornerLength)
    # y = location[1] + math.sin(orientation) * (- viewLength*0.5 - arraySubcornerLength + iy*arraySubcornerLength) + math.cos(orientation) * (viewLength - ix*arraySubcornerLength)
    # x = location[0] + math.cos(orientation) * (- viewLength*0.5 - arraySubcornerLength + iy*arraySubcornerLength) - math.sin(orientation) * (viewLength - ix*arraySubcornerLength)
    return x, y

def get_realtive_angle_to_orientation(angle, orientation):
    relative_angle = angle - orientation - np.pi / 2
    # Adjust the relative angle to be within the range of -π to π
    if relative_angle > np.pi:
        relative_angle -= 2 * np.pi
    elif relative_angle < -np.pi:
        relative_angle += 2 * np.pi
    return relative_angle


def makeInputTensor(encodedMap, egoVelocity, socialInformationOfCarsInView):
        flattened_encodedMap = encodedMap.flatten(start_dim=1).float()
        flattened_egoVelocity = egoVelocity.flatten(start_dim=1).float()
        # flattened_socialInformationOfCarsInView = socialInformationOfCarsInView.flatten(start_dim=1).float()
        flattened_socialInformationOfCarsInView = socialInformationOfCarsInView.flatten(start_dim=1)
        
        combined_tensor = torch.cat((flattened_encodedMap, flattened_egoVelocity, flattened_socialInformationOfCarsInView), dim=1) # Concatenate flattened tensors
        # print(combined_tensor.shape)
        inputTensor = combined_tensor.view(-1,145)
        return inputTensor

def makeInputTensorValidation(encodedMap, egoVelocity, socialInformationOfCarsInView):
        egoVelocity = torch.tensor(egoVelocity)  # Convert egoVelocity to a tensor
        combined_tensor = torch.cat((encodedMap.view(-1,64), egoVelocity.view(-1,1), socialInformationOfCarsInView.view(-1,4*20)), dim=1) # Concatenate flattened tensors
        # print(combined_tensor.shape)
        inputTensor = combined_tensor
        return inputTensor

def calculateVelocity(dx,dy,dt):
    return math.sqrt(dx**2 + dy**2) / dt

def getCovList(velocity):
    covList = np.zeros((30, 2, 2))
    if(velocity != 0):
        for i in range(30):
            # if(velocity != 0):
            covList[i] = np.array([[1+i*0.3, 0.0], [0.0, 1+i*0.3]])
            # else:
            #     covList[i] = np.array([[0.0, 0.0], [0.0, 0.0]])
    return covList