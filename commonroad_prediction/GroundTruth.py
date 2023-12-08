import numpy as np
import math

from configparser import ConfigParser

''' configuration from config.ini file: '''
file = 'commonroad_prediction/config.ini'
config = ConfigParser()
config.read(file)


masterTimesteps = eval(config['var']['masterTimesteps'])

def make_ground_truth_dictionary(scenario):
    """
    Generates the ground truth data for the given scenario and all timesteps.

    Args:
    - scenario (Scenario): The scenario for which to generate ground truth data.

    Returns:
    - positions (dict): A dictionary that contains the positions of all obstacles at each timestep.
                        The keys of the dictionary are the timestep indices, and the values are
                        sub-dictionaries that contain the positions of each obstacle at that timestep.
    """
    ground_truth_dict = {}

    for t in range(0, masterTimesteps):
        ego_states = scenario.obstacle_states_at_time_step(t)
        ground_truth_dict[t] = {}

        for key in ego_states:
            ground_truth_dict[t][key] = {}
            ground_truth_dict[t][key]['position'] = ego_states[key].position
            ground_truth_dict[t][key]['orientation'] = ego_states[key].orientation
            ground_truth_dict[t][key]['velocity'] = ego_states[key].velocity
            ground_truth_dict[t][key]['dx'] = 0
            ground_truth_dict[t][key]['dy'] = 0
            ground_truth_dict[t][key]['absDx'] = 0
            ground_truth_dict[t][key]['absDy'] = 0
    ground_truth_dict = calculate_dx_dy_absDx_absDy_forIn_ground_truth_dict(ground_truth_dict)
    return ground_truth_dict

def calculate_dx_dy_absDx_absDy_forIn_ground_truth_dict(ground_truth_dict):
    """
    Generates ground truth data for a given scenario and all timesteps.

    Args:

    - ground_truth_dict (dict): A dictionary containing the ground truth data for the scenario.

    
    Returns:
    - ground_truth_dict (dict): Updated ground truth data dictionary that includes positions,
        dx, dy, absDx, and absDy for each obstacle at each timestep.
        The keys of the dictionary are the timestep indices, and the values
        are sub-dictionaries containing the updated information for each
        obstacle at that timestep.
    """
    for t in range(0, masterTimesteps):
        for key in ground_truth_dict[t]:
            if t+1 >= masterTimesteps:
                continue
            if key not in ground_truth_dict[t+1]:
                continue

            dx = ground_truth_dict[t+1][key]['position'][0] - \
                ground_truth_dict[t][key]['position'][0]
            dy = ground_truth_dict[t+1][key]['position'][1] - \
                ground_truth_dict[t][key]['position'][1]

            # Desired rotation angle (in radians)
            theta = -1.0 * ground_truth_dict[t][key]['orientation']
            rotation_matrix = np.array([[math.cos(theta), -math.sin(theta)],
                                        [math.sin(theta), math.cos(theta)]])  # Rotation matrix
            original_coordinates = np.array([[dx],
                                            [dy]])  # Original coordinates
            adjusted_coordinates = rotation_matrix.dot(
                original_coordinates)  # Apply rotation
            ground_truth_dict[t][key]['dx'] = adjusted_coordinates[0][0]
            ground_truth_dict[t][key]['dy'] = adjusted_coordinates[1][0]
            ground_truth_dict[t][key]['absDx'] = dx
            ground_truth_dict[t][key]['absDy'] = dy
    return ground_truth_dict        


def get_ground_truth_N_steps_ahead(ground_truth_dict, timestep, obstacle_id, n):
    # ToDo check this!
    max_timestep = max(ground_truth_dict.keys())
    next_positions = []
    for t in range(timestep, min(timestep + n, max_timestep + 1)):
        if obstacle_id in ground_truth_dict[t]:
            next_positions.append(
                ground_truth_dict[t][obstacle_id]['position'])
        else:
            break
    return next_positions

def get_ground_truth_in_velocity_vectors(ground_truth_dict, cr_id):
    """
    Retrieves velocity vectors and absolute velocity vectors for a given obstacle from the ground truth data.

    Args:
    - ground_truth_dict (dict): A dictionary containing the ground truth data for the scenario.
    - cr_id (str): The identifier of the obstacle for which to retrieve the velocity vectors.

    Returns:
    - velocityVectors (ndarray): An array containing the velocity vectors of the obstacle at each timestep.
                                 The shape of the array is (timesteps, 2), where timesteps is the total
                                 number of timesteps in the ground truth data.
    - absVelocityVectors (ndarray): An array containing the absolute velocity vectors of the obstacle at each
                                    timestep. The shape of the array is (timesteps, 2), where timesteps is
                                    the total number of timesteps in the ground truth data.
    """
    velocityVectors = np.ndarray(shape=(masterTimesteps, 2), dtype=float) 
    absVelocityVectors = np.ndarray(shape=(masterTimesteps, 2), dtype=float) 
    # velocityVectors = np.ndarray(shape=(50, 2), dtype=float)
    # absVelocityVectors = np.ndarray(shape=(50, 2), dtype=float)

    for i in range(velocityVectors.shape[0]):
        if cr_id in ground_truth_dict[i]:
            velocityVectors[i] = [ground_truth_dict[i][cr_id]
                                  ['dx'], ground_truth_dict[i][cr_id]['dy']]
            absVelocityVectors[i] = [ground_truth_dict[i][cr_id]
                                     ['absDx'], ground_truth_dict[i][cr_id]['absDy']]
    return velocityVectors, absVelocityVectors




def get_groundTruth_velocity_trajectories_of_scenario(ground_truth_dict):
    """
    Retrieves the output velocity trajectories of all obstacles in the scenario.

    Args:
    - ground_truth_dict (dict): A dictionary containing the ground truth data for the scenario.

    Returns:
    - outputVelocities (dict): A dictionary that contains the output velocity trajectories of all obstacles.
                               The keys of the dictionary are the identifiers of the obstacles, and the values
                               are sub-dictionaries containing the velocity vectors and absolute velocity
                               vectors of each obstacle.
    """
    outputVelocities = {}
    for key in ground_truth_dict[0]:
        outputVelocities.setdefault(key, {})
        velVectors, absVelVectors = get_ground_truth_in_velocity_vectors(
            ground_truth_dict, cr_id=key)
        outputVelocities[key]['velocityVectors'] = velVectors
        outputVelocities[key]['absoluteVelocityVectors'] = absVelVectors
    return outputVelocities

