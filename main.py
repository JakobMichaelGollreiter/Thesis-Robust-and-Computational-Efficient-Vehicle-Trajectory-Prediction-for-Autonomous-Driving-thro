#!/usr/bin/env python3
import random
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import datetime
import math
import time
import os
import moviepy.video.io.ImageSequenceClip
from PIL import Image, ImageFile
from configparser import ConfigParser
ImageFile.LOAD_TRUNCATED_IMAGES = True

# CommonRoad modules:
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad_prediction.prediction_module import PredictionModule
from commonroad_prediction.processing.preprocessing import Preprocessing
from commonroad_prediction.visualization import draw_uncertain_predictions, draw_with_uncertainty, confidence_ellipse, draw_prediction_legend
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet

# My own modules:
from commonroad_prediction.data_generator import DataGenerator
from commonroad_prediction.ScenarioLoader import ScenarioLoader
import commonroad_prediction.visualisation as visualisation
import commonroad_prediction.HelperFunctions as HelperFunctions
import commonroad_prediction.DisplacementError as DisplacementError
import commonroad_prediction.LaneletOperations as LaneletOperations
import commonroad_prediction.GroundTruth as GroundTruth
from commonroad_prediction.NeuralNets import NeuralNets, TrajectoryPredictionNet,Encoder, Decoder, ConvAutoencoder



# for neuralNet
import torch
from torchinfo import summary
from torch.nn.functional import normalize
import numpy as np
import torch.nn as nn
import commonroad_prediction.toPointListConversion as toPointListConversion


# ToDo lanelet.py orientation_by_position_without_assertion() to GitLab

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
socialInformationSize = eval(config['var']['socialInformationSize'])
inference = eval(config['var']['inference'])


def reshape_socialInformationOfCarsInView(socialInformationOfCarsInView):
    socialInformationOfCarsInViewReshaped  = np.zeros((socialInformationSize, 4), dtype=np.float64)  # as a np array
    for i in range(len(socialInformationOfCarsInView)):
        socialInformationOfCarsInViewReshaped[i] = socialInformationOfCarsInView[i]
    return socialInformationOfCarsInViewReshaped

def sort_socialInformationOfCarsInView_by_distance(socialInformationOfCarsInView):
    if len(socialInformationOfCarsInView) <= 1:
        return socialInformationOfCarsInView
    
    mid = len(socialInformationOfCarsInView) // 2 # Divide the list into two halves
    left_half = socialInformationOfCarsInView[:mid]
    right_half = socialInformationOfCarsInView[mid:]

    left_half = sort_socialInformationOfCarsInView_by_distance(left_half)# Recursively sort the two halves
    right_half = sort_socialInformationOfCarsInView_by_distance(right_half)
    
    return merge(left_half, right_half)# Merge the sorted halves

def merge(left_half, right_half):
    merged = []
    left_index = right_index = 0
    
    while left_index < len(left_half) and right_index < len(right_half): # Compare elements from both halves and merge them in sorted order
        if HelperFunctions.distance(left_half[left_index]) <= HelperFunctions.distance(right_half[right_index]):
            merged.append(left_half[left_index])
            left_index += 1
        else:
            merged.append(right_half[right_index])
            right_index += 1
    
    while left_index < len(left_half): # Append any remaining elements from the left half
        merged.append(left_half[left_index])
        left_index += 1
    
    while right_index < len(right_half): # Append any remaining elements from the right half
        merged.append(right_half[right_index])
        right_index += 1
    
    return merged

def initialize_one_datapoint():
    reachableRoadFromCurrentLaneletInView = np.ndarray(shape=arrayShape, dtype=float, order='F')
    roadAroundCurrentLaneletInView = np.ndarray(shape=arrayShape, dtype=float, order='F')
    groundTruthTrajectoryInView = np.ndarray(shape=arrayShape, dtype=int, order='F')
    socialInformationOfCarsInView = np.zeros((socialInformationSize, 4), dtype=np.float64)  # as a np array

    reachableRoadFromCurrentLaneletInView.fill(0.0)
    roadAroundCurrentLaneletInView.fill(0.0)
    groundTruthTrajectoryInView.fill(0)
    return reachableRoadFromCurrentLaneletInView, roadAroundCurrentLaneletInView, groundTruthTrajectoryInView, socialInformationOfCarsInView


def makeInputFeaturesPlusGroundTruth(scenario, time_begin, ground_truth_dict, groundTruthVelocityTrajectories, inputFeaturesPlusGroundTruth):
    """
    This function, generates the input features(mapInView, egoVelocity, socialInformationOfCarsInView) for all vehicles in that timestep and stores them in the dictionary inputFeaturesPlusGroundTruth.

    Inputs:
    - `scenario`: Represents the scenario or environment in which the vehicles are moving.
    - `time_begin`: The current time step for processing.
    - `ground_truth_dict`: A dictionary containing ground truth info about obstacles at different time steps.
    - `groundTruthVelocityTrajectories`: A dictionary containing velocity trajectories of obstacles.
    - `inputFeaturesPlusGroundTruth`: A dictionary being populated with information for each obstacle.

    Processing:
    The function iterates over each `cr_id` (obstacle ID) in `ground_truth_dict` for current timestep. For each obstacle:
    - The obstacle's occupancy info and orientation are extracted.
    - Ego vehicle's states at the time step are obtained.
    - Ego vehicle's velocity is extracted.
    - Current obstacle location is obtained.
    - Driveable lanelets from the obstacle's position are determined(important for creating mapInView feature).
    - The function `fill_all_information_tensors_of_one_datapoint` actually fills the mapInView, egoVelocity, socialInformationOfCarsInView tensors.
    - Ground truth trajectory is extracted, manipulated, and approximated if needed.
    - Gathered info is stored in `inputFeaturesPlusGroundTruth` under obstacle ID and time step.

    Output:
    The function returns the `inputFeaturesPlusGroundTruth` dictionary, containing ego velocity, map view, social info, ground truth trajectory, etc., for each obstacle at the specified time step.
    """

    LaneletNetwork = scenario.lanelet_network
    for cr_id in ground_truth_dict[time_begin]:
        obstacle = scenario.obstacle_by_id(cr_id)

        obstacle_occupancy = obstacle.occupancy_at_time(time_begin) # get current occupancy of vehicle
        orientation_of_vehicle = obstacle_occupancy.shape.orientation
        ego_states = scenario.obstacle_states_at_time_step(time_begin)
        egoVelocity = ego_states[cr_id].velocity # get the velocity of the Ego Vehicle

        location =  obstacle_occupancy.shape.center[0], obstacle_occupancy.shape.center[1] # get current position of vehicle
        position = np.array([location[0], location[1]])
        merged_successor_lanelets, closest_lanelet = LaneletOperations.get_all_lanelets_with_different_successor_options(position, orientation_of_vehicle, LaneletNetwork) # merged_successor_lanelets are the lanelets and succesors that are driveable from the current position 

        mapInView, groundTruthTrajectoryInView, socialInformationOfCarsInView = fill_all_information_tensors_of_one_datapoint(location, scenario, cr_id, ego_states, orientation_of_vehicle, time_begin, ground_truth_dict, merged_successor_lanelets, LaneletNetwork, closest_lanelet, groundTruthVelocityTrajectories)

        inputFeaturesPlusGroundTruth.setdefault(cr_id, {})
        groundTruthTrajectory = groundTruthVelocityTrajectories[cr_id]['velocityVectors'][time_begin:time_begin+30]
        groundTruthTrajectory = np.insert(groundTruthTrajectory, 2, 1, axis=1)

        if ((time_begin + 30) > (masterTimesteps-1)):
            groundTruthTrajectory = approximate_groundTruthTrajectroy_if_out_of_time(groundTruthTrajectory)
        inputFeaturesPlusGroundTruth[cr_id][time_begin] = {'egoVelocity': egoVelocity, 'mapInView': mapInView, 'socialInformationOfCarsInView': socialInformationOfCarsInView, 'groundTruthTrajectoryInView': groundTruthTrajectoryInView, 'groundTruthTrajectory': groundTruthTrajectory}
    return inputFeaturesPlusGroundTruth

def approximate_groundTruthTrajectroy_if_out_of_time( groundTruthTrajectory):
    groundTruthTrajectory = np.delete(groundTruthTrajectory, -1, axis=0) # ToDo check if this is needed?
    lastDx = groundTruthTrajectory[-1][0]
    padWidth = [[lastDx, 0, -1]]* (30-(len(groundTruthTrajectory))) # keep the lastDx as dx and create a straight trajectory for the rest of the timesteps
    groundTruthTrajectory = np.concatenate((groundTruthTrajectory, padWidth), axis=0)
    return groundTruthTrajectory

def get_socialInteractionsBetweenVehicles_to_ego(relativePositionAndVelocityToEgo, carsInProximity, location, orientation_of_vehicle, ego_states, scenario):
    """
    This function calculates and gathers information about the social interactions between surrounding vehicles and the ego vehicle.

    Inputs:
    - `relativePositionAndVelocityToEgo`: A matrix to store relative position and velocity information of surrounding vehicles to the ego.
    - `carsInProximity`: List of obstacle IDs representing vehicles in proximity to the ego vehicle.
    - `location`: Current location of the ego vehicle.
    - `orientation_of_vehicle`: Orientation of the ego vehicle.
    - `ego_states`: Ego vehicle's states at the specified time step.
    - `scenario`: The scenario or environment in which vehicles are moving.

    Processing:
    - States of surrounding vehicles are initialized.
    - Counter for filling relative position and velocity information is set(max 20).
    - For each vehicle in proximity (`carsInProximity`):
        1 Vehicle's occupancy and velocity information are extracted.
        2 Relative position and angle of the surrounding vehicle with respect to the ego vehicle are calculated.
        3 Relative velocity information is calculated.
        4 Relative position and velocity information is stored in the `relativePositionAndVelocityToEgo` matrix.

    Output:
    - The function returns the `relativePositionAndVelocityToEgo` matrix, containing relative position and velocity information for surrounding vehicles in relation to the ego vehicle.
    """
    statesOfSurrouningVehicles = {}
    relativePositionAndVelocityToEgoFillCounter = 0
    for car in carsInProximity:
        carObstacleID = scenario.obstacle_by_id(car)
        carObstacle_occupancy = carObstacleID.occupancy_at_time(ego_states[carObstacleID.obstacle_id].time_step)
        velocity = ego_states[carObstacleID.obstacle_id].velocity
        statesOfSurrouningVehicles[carObstacleID.obstacle_id] = [carObstacle_occupancy.shape.center, carObstacle_occupancy.shape.orientation, velocity]
        angle = statesOfSurrouningVehicles[car][1]

        absolutePositionToEgo = [-(location[1] - carObstacle_occupancy.shape.center[1]), location[0] - carObstacle_occupancy.shape.center[0]]  # Ego -> car
        relativePositionToEgo = HelperFunctions.calculate_relativePositionToEgo(absolutePositionToEgo, orientationOfEgo=orientation_of_vehicle)
        relativeAngleToEgo = HelperFunctions.get_realtive_angle_to_orientation(angle, orientation_of_vehicle)
        relativeVelocityToEgo = HelperFunctions.calculate_relativeDxDyVelocityToEgo(velocity, relativeAngleToEgo)

        relativePositionAndVelocityToEgo[relativePositionAndVelocityToEgoFillCounter][0] = relativePositionToEgo[0]
        relativePositionAndVelocityToEgo[relativePositionAndVelocityToEgoFillCounter][1] = relativePositionToEgo[1]
        relativePositionAndVelocityToEgo[relativePositionAndVelocityToEgoFillCounter][2] = relativeVelocityToEgo[0]
        relativePositionAndVelocityToEgo[relativePositionAndVelocityToEgoFillCounter][3] = relativeVelocityToEgo[1]
        relativePositionAndVelocityToEgoFillCounter += 1
    return relativePositionAndVelocityToEgo    

def fill_all_information_tensors_of_one_datapoint(location, scenario, cr_id, ego_states, orientation_of_vehicle, time_begin, ground_truth_dict, merged_successor_lanelets, LaneletNetwork, closest_lanelet, groundTruthVelocityTrajectories):

    """
    Inputs:
    - `location`: Current location of the obstacle.
    - `scenario`: The scenario or environment in which the obstacle is moving.
    - `cr_id`: ID of the current obstacle.
    - `ego_states`: Ego vehicle's states at the specified time step.
    - `orientation_of_vehicle`: Orientation of the obstacle.
    - `time_begin`: current time step 
    - `ground_truth_dict`: Dictionary containing ground truth info about obstacles at different time steps.
    - `LaneletNetwork`: The network of lanelets in the scenario.
    - `merged_successor_lanelets`: Lanelets and successors that are driveable from the current position.
    - `closest_lanelet`: Closest lanelet to the obstacle's current position.
    - `groundTruthVelocityTrajectories`: Velocity trajectories of obstacles.

    Processing:
    - Orientation of the vehicle is adjusted.
    - Obstacles in proximity are obtained using `get_obstacles_in_radius` function.
    - Lanelets and successors are flattened into a list.
    - Various matrices are initialized for reachable roads, road around current lanelet, ground truth trajectory, and social information.
    - Social interactions between vehicles in view are gathered using `get_socialInteractionsBetweenVehicles_to_ego`.
    - Social information is sorted by distance to ego vehicle.
    - (optional)Looping over the view field and time steps, the ground truth trajectory array is filled.
    - Lanelets are found based on position, and matrices for reachable and possible lanes are filled.
    - The mapInView feature is constructed by combining reachable and possible road information.

    Output:
    - The function returns three main outputs:
    - `mapInView`: The local view map of the ego vehicle with 0.5 indicating a road, 1.0 indicating reachable (driveable) positions, and 0.0 indicating non-road positions.
    - `socialInformationOfCarsInView`: Social information about other cars in view of the ego vehicle.
    - (optional)`groundTruthTrajectoryInView`: Ground truth trajectory in view, indicating where the obstacle is going to move.
    """
    orientation_of_vehicle = orientation_of_vehicle - math.pi/2
    carsInProximity = Preprocessing.get_obstacles_in_radius(scenario, cr_id, ego_state=ego_states[cr_id], radius=oneDimension*arraySubcornerLength)  # ToDo: make this radius dependent on velocity
    flat_list_merged_successor_lanelets = [item for sublist in merged_successor_lanelets for item in sublist]
    
    reachableRoadFromCurrentLaneletInView, roadAroundCurrentLaneletInView, groundTruthTrajectoryInView, socialInformationOfCarsInView = initialize_one_datapoint()
    socialInformationOfCarsInView = get_socialInteractionsBetweenVehicles_to_ego(socialInformationOfCarsInView, carsInProximity, location, orientation_of_vehicle, ego_states, scenario)
    socialInformationOfCarsInView = sort_socialInformationOfCarsInView_by_distance(socialInformationOfCarsInView) # merge sort the socialInformationOfCarsInView by distance to ego

    for iy in range(reachableRoadFromCurrentLaneletInView.shape[1]): # loop over the 2 dimensional array(e.g. 19x19, 49x49) in the view field of the ego vehicle
        for ix in range(reachableRoadFromCurrentLaneletInView.shape[0]):
            x, y = HelperFunctions.calculate_position_in_oriented_coordinate_system(orientation_of_vehicle, location, ix, iy)
            posX = ground_truth_dict[0][cr_id]['position'][0]
            posY = ground_truth_dict[0][cr_id]['position'][1]
            for t in range(masterTimesteps):
                time = time_begin + t # ToDo What is this?
                if t > 0:
                    posX += groundTruthVelocityTrajectories[cr_id]['absoluteVelocityVectors'][t-1][0]
                    posY += groundTruthVelocityTrajectories[cr_id]['absoluteVelocityVectors'][t-1][1]
                if (HelperFunctions.ground_truth_is_current_position_a_hit(x, y, posX, posY)):
                    groundTruthTrajectoryInView[ix, iy] = 1  # fill ground truth array with 1 if the current position is a hit in all possible timesteps
            lanes = LaneletNetwork.find_lanelet_by_position([(x, y)])
            flat_list_lanes = [item for sublist in lanes for item in sublist]
            reachableRoadFromCurrentLaneletInView = LaneletOperations.fillAllReachableLanes(flat_list_lanes, flat_list_merged_successor_lanelets, closest_lanelet, reachableRoadFromCurrentLaneletInView, ix, iy) # filled with with 0.5 if this position reachable(driveable) road from the current lanelet 
            roadAroundCurrentLaneletInView = LaneletOperations.fillAllPossibleLanes(flat_list_lanes, roadAroundCurrentLaneletInView, ix, iy) # with 1 if this position is a road at all(e.g. opposite track of road)
    mapInView = np.where(reachableRoadFromCurrentLaneletInView != 0.0, reachableRoadFromCurrentLaneletInView, roadAroundCurrentLaneletInView) # this is the local view map of the ego vehicle with 0.5 if the position is a road and 1.0 if the position is reachable(driveable) from the current lanelet and 0.0 if this is not a road
    return mapInView, groundTruthTrajectoryInView, socialInformationOfCarsInView


def create_current_prediction(scenario, pred, time_step):
    """
    Given a scenario, a PredictionModule instance and a time_step, returns the predictions
    for the ego state at that time_step.

    Args:
    - scenario (Scenario): the scenario instance
    - pred (PredictionModule): an instance of the PredictionModule class
    - time_step (int): the time step for which to generate the predictions

    Returns:
    - predictions (list): a list of predictions for the ego state at the given time step
    """
    # I want to get all ego_states for the current time step and then iterate over them to get the predictions for each of them
    ego_states = scenario.obstacle_states_at_time_step(time_step)
    ego_key = min(ego_states.keys())
    ego_state = ego_states[ego_key]
    predictions = pred.main_prediction(ego_state=ego_state, sensor_radius=1000, t_list=[])
    return predictions



def print_out_current_prediction(predictions):
    for cr_id in predictions.keys():
        print("ID: ", cr_id)
        
        prediction_for_obstacle = predictions[cr_id] # get the prediction for obstacle with cr_id=cr_id
        
        pos_list_for_obstacle = prediction_for_obstacle['pos_list'] # get the pos_list and cov_list values for obstacle with cr_id
        cov_list_for_obstacle = prediction_for_obstacle['cov_list']
        print("\n Position list for obstacle with ID: ",
              cr_id, "\n", pos_list_for_obstacle)
        print("\n Covariance list for obstacle with ID: ",
              cr_id, "\n", cov_list_for_obstacle, "\n\n")


def getInputFeaturesPlusGroundTruth(scenario, image_folder, ground_truth_dict, dt, SCENARIO_PATH, groundTruthVelocityTrajectories):
    """
    returns input features and ground truth in data generation (in inference also predictions are made at each timestep and visualized(with buildImagesAndVideo))

    Returns:
        dict: A dictionary containing input features and ground truth for all cars and timesteps.

    This function processes a scenario by performing the following steps:
    1. (During inference) Load necessary neural networks and normalization parameters if inference is enabled.
    2. Iterate over time steps in the scenario and generate input features.
    3. (During inference) Generate predictions using a neural network or a different prediction module.
    4. (During inference) Calculate ADE and FDE for the predictions.
    5. Optionally generate images and visualizations.
    6. (During inference)Print mean ADE and FDE values.
    7. Return the dictionary containing input features and ground truth.
    """

    inputFeaturesPlusGroundTruth = {} # this is the dictionary that contains (the local view map,  groundTruthTrajectories, own Velocity and socialInteractionsBetweenVehicles ) for all cars and all timesteps
    adeList, fdeList = [],[]
    if (inference == True): 
        learntCNN, model = NeuralNets().loadNeuralNets()
        egoVelocityNorm, socialInformationOfCarsInViewNorm, encodedMapNorm, groundTruthNorm = loadNormsForNeuralNet()
    for i in range(0, masterTimesteps):
        if ((masterTimesteps-i) <= (cutOffTimeSteps)): # if there is to little time in scenario left for making a valid groundTruthTrajectory, break
            break
        inputFeaturesPlusGroundTruth = makeInputFeaturesPlusGroundTruth(scenario, i, ground_truth_dict, groundTruthVelocityTrajectories, inputFeaturesPlusGroundTruth) # generate input features for all vehicles in that timestep
        pred = PredictionModule(scenario=scenario, timesteps=masterTimesteps, dt=dt, detection_method="notFOV")  # detection_methode = 'notFOV' makes sure that the prediction is made for all obstacles in the scenario in a certain radius (No Occlusion)
        if (inference == True):
            groundTruthPositionForCurrentTimeStep = ground_truth_dict[i] # only get the current(time_step = i) position x,y and orientation  
            predictions = createPredictionsFromNeuralNet(inputFeaturesPlusGroundTruth, i,learntCNN, model, egoVelocityNorm, socialInformationOfCarsInViewNorm, encodedMapNorm, groundTruthNorm, groundTruthPositionForCurrentTimeStep, dt)
        else:
            predictions = create_current_prediction(scenario=scenario, pred=pred, time_step=i)  # get the predictions for the current time step of scenario # Here one can input the traind neural prediction model
        ade, fde = DisplacementError.calculate_ADE_and_FDE(predictions, ground_truth_dict, timestep=i)  # calculate the ADE and FDE for the current time step Important that ADE and FDE are normally 5 seconds long e.g. 50 timesteps
        adeList.append(ade)
        fdeList.append(fde)
        # comprehensiveTrainingData = combine_comprehensiveTrainingData_and_socialInteractionsBetweenVehicles(socialInteractionsBetweenVehicles, comprehensiveTrainingData)
        if (buildImagesAndVideo == True):
            frame = visualisation.visualisation_scenario_prediction_one_timestep(i, i+1, scenario, predictions, ground_truth_dict, SCENARIO_PATH, adeList, fdeList, inputFeaturesPlusGroundTruth) # ToDo maybe just i, i
            image_file_name = f"frame_{i}.png"
            frame.savefig(os.path.join(image_folder, image_file_name), dpi=300) # dpi is for the quality/resolution of the image
    print(f'\nmean ade over all vehicles and timesteps = {np.mean(adeList)}')        
    print(f'mean fde over all vehicles and timesteps = {np.mean(fdeList)}\n')        
    return inputFeaturesPlusGroundTruth


def loadNormsForNeuralNet(): 
    egoVelocityNorm = torch.load('commonroad_prediction/NormsTest/eviEgoVelocityNorm39.pt')
    socialInformationOfCarsInViewNorm = torch.load('commonroad_prediction/NormsTest/eviSocialInformationOfCarsInViewNorm39.pt')
    encodedMapNorm = torch.load('commonroad_prediction/NormsTest/eviEncodedMapNorm39.pt')
    groundTruthNorm = torch.load('commonroad_prediction/NormsTest/eviRelGroundTruthNorm39.pt')
    # encodedMapNorm = torch.fill_(encodedMapNorm, 1.0)
    return egoVelocityNorm, socialInformationOfCarsInViewNorm, encodedMapNorm, groundTruthNorm

def createPredictionsFromNeuralNet(inputFeaturesPlusGroundTruth, time_step, learntCNN, model, egoVelocityNorm, socialInformationOfCarsInViewNorm, encodedMapNorm, groundTruthNorm, groundTruthPositionForCurrentTimeStep, dt):
    """
    Generate trajectory predictions for all vehicles present at this time_step.
    Args:
        inputFeaturesPlusGroundTruth (dict): Input features and ground truth data for multiple cars.
        time_step (int): The current time step for which predictions are to be made.
        learntCNN (torch.nn.Module): The learned Convolutional Neural Network (CNN) used for encoding the map.
        model (torch.nn.Module): The neural network model used for trajectory prediction.
        egoVelocityNorm (float): Normalization factor for ego velocities.
        socialInformationOfCarsInViewNorm (numpy.ndarray): Normalization factors for social information of cars in view.
        encodedMapNorm (float): Normalization factor for the encoded map.
        groundTruthNorm (float): Normalization factor for ground truth values.
        groundTruthPositionForCurrentTimeStep (dict): Ground truth position and orientation data for cars at the current time step.
        dt (float): Time step duration used for converting velocity to positions.
    Returns:
        dict: Trajectory predictions for multiple cars.
    Note:
        - Does not use the groundTruthTrajectory for predictions.
        - The dimensions (length and width) of the car shape are set to standard values and are not crucial for the prediction.
    """
    predictions = {}
    for cr_id in inputFeaturesPlusGroundTruth.keys(): 
        if(inputFeaturesPlusGroundTruth[cr_id].get(time_step) is None): # if car not in scenario anymore,
            continue
        output = getOutputTrajectoryFromNeuralNet(inputFeaturesPlusGroundTruth,cr_id,time_step,learntCNN, model, egoVelocityNorm, socialInformationOfCarsInViewNorm, encodedMapNorm, groundTruthNorm)
        velocity = inputFeaturesPlusGroundTruth[cr_id][time_step]['egoVelocity']
        initialOrientation = groundTruthPositionForCurrentTimeStep[cr_id]['orientation']
        outputAsGlobalPointList = toPointListConversion.convertOutputTrajectoryToGloabalPointList(output, cr_id, groundTruthPositionForCurrentTimeStep,viewLength)
        covList = HelperFunctions.getCovList(velocity)
        shape = {'length': 4.5, 'width': 2.0} # arbitary dimensions for the car (not important for the prediction)
        globalOrientationList, velocityList = toPointListConversion.getOrientationListAndVelocityList(output, initialOrientation, dt)
        predictions[cr_id] = {'pos_list': outputAsGlobalPointList, 'cov_list': covList, 'orientation_list': globalOrientationList, 'v_list': velocityList, 'shape': shape}
    return predictions    


def getOutputTrajectoryFromNeuralNet(inputFeaturesPlusGroundTruth,cr_id,time_step,learntCNN, model, egoVelocityNorm, socialInformationOfCarsInViewNorm, encodedMapNorm, groundTruthNorm):
    """
    Generate a trajectory prediction for the current car using the given input features.
    Args:
        mapInView (numpy.ndarray): 2D array representing the map information in the car's view.
        egoVelocity (float): The velocity of the current car.
        socialInformationOfCarsInView (numpy.ndarray): 2D array representing the social information of cars in the current car's view.
    Returns:
        numpy.ndarray: The output trajectory prediction for the current car.
    Note:
        - This function utilizes a learned Convolutional Neural Network (CNN) encoder to process the input map.
        - The input map, egoVelocity, and socialInformationOfCarsInView are normalized before processed in the networks
        - Even though present, the groundTruthTrajectory is not accessed this function.
    """
    dataCurrentCarTimeStep = inputFeaturesPlusGroundTruth[cr_id][time_step]
    input_sizeX = dataCurrentCarTimeStep['mapInView'].shape[0]
    input_sizeY = dataCurrentCarTimeStep['mapInView'].shape[1]
    image = torch.tensor(dataCurrentCarTimeStep['mapInView']).float().view(-1, input_sizeX , input_sizeY) 
    image = image.unsqueeze(1)
    encodedMap = learntCNN.encoder.forward(image).view(-1,64) # Output of encoder dimensionality reduction

    egoVelocity = torch.tensor(dataCurrentCarTimeStep['egoVelocity'])
    egoVelocity = torch.where(egoVelocity != 0, egoVelocity / egoVelocityNorm, torch.zeros_like(egoVelocity)) # normalizing the egoVelocity
    socialInformationOfCarsInView = torch.tensor(dataCurrentCarTimeStep['socialInformationOfCarsInView'])
    socialInformationOfCarsInView = torch.where(socialInformationOfCarsInView != 0, socialInformationOfCarsInView / socialInformationOfCarsInViewNorm.view(20,4), torch.zeros_like(socialInformationOfCarsInView)).view(80) # normalizing the socialInformationOfCarsInView
    inputTensor = HelperFunctions.makeInputTensorValidation(encodedMap, egoVelocity, socialInformationOfCarsInView).view(-1,145).to(torch.float32) # convert to float32
    prediction = model.forward(inputTensor)  # Forward pass -> prediction

    ### Postprocessing --> Only positive x-driving direction allowed
    prediction = prediction.view(30,2) # Separating the two columns
    columnx = prediction[:, 0]
    columny = prediction[:, 1]
    
    abs_columnx = torch.abs(columnx) # Computing the absolute values for x column (only positive driving direction allowed) -> more stable training
    prediction = torch.stack((abs_columnx, columny), dim=1).view(30,2) # Combining the results back into a tensor with shape (30, 2)
    prediction = prediction.view(30*2)
    ###


    prediction = torch.where(prediction != 0, prediction * groundTruthNorm, torch.zeros_like(prediction)) # denormalizing the output
    return prediction

def main():  # for data generation
    scenarioLoader = ScenarioLoader()
    generator = DataGenerator()

    folderPath = scenarioLoader.get_scenario_folder_path()
    image_folder = ''
    folderContents = os.listdir(folderPath)
    for filename in folderContents[0:]: # 160
        try: # for creation of dataset (if one scenario fails -> except throws and data generation continues with next scenario in folder)
            start_time = time.time()  # Record the start time
            # ToDo: add automatic timestp detection from scenario (How long is the scenario?)

            SCENARIO_PATH, oneScenario = scenarioLoader.get_specific_scenario(), True
            # SCENARIO_PATH, oneScenario = os.path.join(folderPath, filename), False # get files in order from folder, get_scenario_folder_path()
            # SCENARIO_PATH, oneScenario = scenarioLoader.get_random_training_scenario(), True

            scenario, planning_problem_set = CommonRoadFileReader(SCENARIO_PATH).open()  # read in the scenario and planning problem set
            dt = scenario.dt  # time step size

            ground_truth_dict = GroundTruth.make_ground_truth_dictionary(scenario) # truth dictionary contains (position, orientation, velocity, dx, dy) of every cr_id in every timestep 
            groundTruthVelocityTrajectories = GroundTruth.get_groundTruth_velocity_trajectories_of_scenario(ground_truth_dict) # output velocity trajectories of every car (This is used as groundTruth for training the neural net)
            
            if (buildImagesAndVideo == True):
                image_folder = visualisation.initialise_prediction_image_folder() # create folder that contains the images of every timestep of the scenario including the prediction and ground truth trajectories
            inputFeaturesPlusGroundTruth = getInputFeaturesPlusGroundTruth(scenario, image_folder, ground_truth_dict, dt, SCENARIO_PATH, groundTruthVelocityTrajectories) # this is the dictionary that contains (mapInView, egoVelocity, socialInformationOfCarsInView features and ground truth) for all cars and all timesteps

            if (buildImagesAndVideo == True):
                visualisation.make_trajectory_video(image_folder, SCENARIO_PATH)

            end_time = time.time()  # Record the start time
            print("\n START TIME: ", start_time, "  END TIME: ", end_time)
            print("Total time: ", end_time - start_time)
            if (inference != True):
                generator.make_input_and_output_csv_files(inputFeaturesPlusGroundTruth, SCENARIO_PATH)
            if(inference == True or oneScenario == True):
                break

        except Exception as e: # Handle the exception for creation of dataset
            print(f"Error processing file: {filename}. {str(e)}")
            continue


if __name__ == "__main__":
    main()
