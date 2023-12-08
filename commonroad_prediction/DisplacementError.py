import numpy as np

import commonroad_prediction.GroundTruth as GroundTruth

def calculate_ADE_and_FDE(predictions, ground_truth_dict, timestep):
    """
    Calculates the Average Displacement Error (ADE) and Final Displacement Error (FDE) for the given predictions
    and ground truth data. The ADE and FDE are calculated for each obstacle and then averaged over all obstacles.
    A time horizon of 30 timesteps is used for the ADE and FDE calculation.

    Args:
    - predictions (dict): A dictionary containing the predictions for each obstacle.
    - ground_truth_dict (dict): A dictionary containing the ground truth data for the scenario.
    - timestep (int): The timestep at which to calculate the ADE and FDE.

    Returns:
    - ade (float): The Average Displacement Error for all obstacles at the specified timestep.
    - fde (float): The Final Displacement Error for all obstacles at the specified timestep.
    """
    
    ade_list = []
    fde_list = []
    prediction_length_list = []

    for obstacle_id in predictions.keys():
        pos_list = predictions[obstacle_id]['pos_list'] # position list is 30 timesteps long

        ground_truth_fifty_steps_ahead = GroundTruth.get_ground_truth_N_steps_ahead(
            ground_truth_dict, timestep + 1, obstacle_id, n=30)
        # for i in range(min(len(predictions[obstacle_id]['pos_list']),len(ground_truth_fifty_steps_ahead))):
        #     pos = "Predicted position {}: {}".format(i, predictions[obstacle_id]['pos_list'][i])
        #     gt = "  Ground truth position {}: {}".format(i, ground_truth_fifty_steps_ahead[i])
        #     print("{:<30}{}".format(pos, gt))

        # print("\nlengPos ", len(pos_list), "lengGround ",len(ground_truth_fifty_steps_ahead),"\n")

        # initialize an empty list to store RMSE values
        rootDeAtTimeStepPositionList = []
        # loop over the time steps
        # ToDo check if this is okay to cut the lenght of the lists!
        prediction_length = min(len(pos_list), len(
            ground_truth_fifty_steps_ahead))
        prediction_length_list.append(prediction_length)
        for i in range(prediction_length):
            # calculate the RMSE for this time step
            # rmse = np.sqrt(np.mean((pos_list[i] - ground_truth_fifty_steps_ahead[i])**2))
            x_y_error_squared = (pos_list[i][0] - ground_truth_fifty_steps_ahead[i][0])**2 + (
                pos_list[i][1] - ground_truth_fifty_steps_ahead[i][1])**2
            rootDeAtTimeStepPosition = np.sqrt(x_y_error_squared)
            rootDeAtTimeStepPositionList.append(rootDeAtTimeStepPosition)

        if len(rootDeAtTimeStepPositionList) > 0:
            fdeOneVehicle = rootDeAtTimeStepPositionList[-1] # last position of x_y_error_squared_list is the position for fde
            adeOneVehicle = np.mean(rootDeAtTimeStepPositionList)
        else:
            # Handle the case where the list is empty e.g last timestep
            adeOneVehicle = 0  # or whatever default value you want
            fdeOneVehicle = 0  # or whatever default value you want

        ade_list.append(adeOneVehicle)
        fde_list.append(fdeOneVehicle)
    mean_prediction_length = np.mean(prediction_length_list)
    # ToDo check if this is okay to divide by mean_prediction_length
    # ade = np.sum(ade_list)/(predictions.keys().__len__()* mean_prediction_length)
    ade = np.sum(ade_list)/(predictions.keys().__len__())
    print("mean_prediction_length ", mean_prediction_length)
    fde = np.mean(fde_list)
    return ade, fde

