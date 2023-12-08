# standard imports
import os

# third-party imports
import configparser
import numpy as np
from typing import Type

# internal imports
from commonroad_prediction.processing.preprocessing import Preprocessing
from commonroad_prediction.processing.postprocessing import Postprocessing

# globals
current_directory_path = os.path.dirname(__file__)
path_config_file = os.path.join(
    current_directory_path, 'config/parameters.ini')


# ToDo: Allow for dynamic adjustment of covariance matrix (adapt to max width of street)
class PredictionModule:

    def __init__(
        self,
        scenario,
        timesteps: int,
        dt: int,
        detection_method: str = "FOV"
    ):

        # variables given by the scenario
        self.dt = dt
        self.scenario = scenario

        # prediction specific variables
        self.prediction_horizon = timesteps
        self.detection_method = detection_method

        # configuration
        config = configparser.ConfigParser()
        config.read(path_config_file)
        self.pred_config = config['PREDICTION']
        self.ego_id = config['COMMONROAD']['EGO_ID']

    def update_scenario(self, scenario):
        self.scenario = scenario

    def predict_trajectory(self, object: 'PredictionObject'):

        if object.prediction_type.type == "Linear":
            self._predict_linear_trajectory(object)
        elif object.prediction_type.type == "Ground_truth":
            self._predict_ground_truth(object)
        elif object.prediction_type.type == "Lanebased":
            self._predict_lane_based_trajectory(object)

    def _predict_linear_trajectory(self, object: 'PredictionObject'):

        if object.velocity == 0:
            fut_pos = [object.position for _ in range(self.prediction_horizon)]
            fut_cov = [[[0.0, 0.0], [0.0, 0.0]]
                       for _ in range(self.prediction_horizon)]
            object.prediction = {'state_list': np.array(
                fut_pos), 'cov_list': np.array(fut_cov)}
            return

        fut_cov = []
        fut_pos = []
        v = object.get_cartesian_velocities()
        for index in range(self.prediction_horizon):

            x = object.position[0] + v["x"] * self.dt * (index+1)
            y = object.position[1] + v["y"] * self.dt * (index+1)
            position = np.array([x, y])

            fut_pos.append(position)
            fut_cov.append([[1.0 + (index * 0.3), 0.0],
                            [0.0, 1.0 + (index * 0.3)]])

        fut_cov = np.array(fut_cov)
        fut_pos = np.array(fut_pos)

        object.prediction = {'state_list': fut_pos, 'cov_list': fut_cov}

    def _predict_ground_truth(self, object: 'PredictionObject'):

        print("GROUND TRUTH")
        trajectory_segment = object.reference_trajectory[1:self.prediction_horizon + 1]

        fut_pos = []
        for state in trajectory_segment:
            fut_pos.append(state.position)

        fut_cov = []
        for index in range(len(trajectory_segment)):
            fut_cov.append([[1.0 + (index * 0.3), 0.0],
                            [0.0, 1.0 + (index * 0.3)]])

        fut_cov = np.array(fut_cov)
        fut_pos = np.array(fut_pos)

        object.prediction = {'state_list': fut_pos, 'cov_list': fut_cov}

    def _predict_lane_based_trajectory(self, object: 'PredictionObject'):

        if len(object.prediction_trajectories) == 0:
            self._predict_linear_trajectory(object)
            return
        else:
            # TODO: check how to handle this better
            trajectory_segment = object.prediction_trajectories[0][1:self.prediction_horizon + 1]

        fut_pos = []
        for state in trajectory_segment:
            fut_pos.append(state.position)

        fut_cov = []
        for index in range(len(trajectory_segment)):
            fut_cov.append([[1.0 + (index * 0.3), 0.0],
                            [0.0, 1.0 + (index * 0.3)]])

        fut_cov = np.array(fut_cov)
        fut_pos = np.array(fut_pos)

        object.prediction = {'state_list': fut_pos, 'cov_list': fut_cov}

    def main_prediction(self, ego_state, sensor_radius, t_list):

        # get relevant objects
        if self.detection_method == "FOV":
            visible_obstacles = Preprocessing.get_obstacles_in_view(
                self.scenario,
                ego_id=self.ego_id,
                ego_state=ego_state,
                fov=float(self.pred_config["FIELD_OF_VIEW"]),
                radius=sensor_radius
            )
        else:
            visible_obstacles = Preprocessing.get_obstacles_in_radius(
                self.scenario,
                ego_id=self.ego_id,
                ego_state=ego_state,
                radius=sensor_radius
            )

        # convert the prediction object
        prediction_objects = []
        for obstacle_id in visible_obstacles:
            obj = Preprocessing.convert_to_prediction_object(
                self.scenario, self.pred_config, obstacle_id, ego_state.time_step
            )
            if obj != None:
                prediction_objects.append(obj)

        # perform prediction
        predictions = {}
        for obstacle in prediction_objects:

            # calculate reference path
            obstacle.calculate_prediction_trajectories(
                self.scenario, self.pred_config, self.dt)

            # update velocity
            # obstacle.velocity = min(
            #     obstacle.velocity + obstacle.max_acceleration,
            #     obstacle.max_velocity
            # )

            # update position
            # current_state = obstacle.occupancy_at_time(t)
            # current_state = obstacle.reference_trajectory[t]
            t = ego_state.time_step
            cr_obj = self.scenario.obstacle_by_id(obstacle.cr_id)
            obstacle.position = cr_obj.occupancy_at_time(t).shape.center
            # obstacle.trajectory.append(ego_state)
            # obstacle.orientation = np.rad2deg(obstacle.orientation)

            # predict trajectory
            # TODO: use rank to order different predictions (return more then one)
            self.predict_trajectory(obstacle)
            predictions[obstacle.cr_id] = {
                'pos_list': obstacle.prediction['state_list'], 'cov_list': obstacle.prediction['cov_list']}

            # update commonroad state
            # updated_trajectory = Trajectory(init_time_step, obstacle.trajectory + obstacle.prediction['state_list'])
            # if obstacle.type == "car":
            #     updated_movement = TrajectoryPrediction(updated_trajectory, Rectangle(obstacle.dimensions[1], obstacle.dimensions[0]))
            # else:
            #     updated_movement = TrajectoryPrediction(updated_trajectory, Circle(obstacle.radius))
            # cr_ref.prediction = updated_movement

        predictions = Postprocessing.add_orientation_velocity_and_shape_to_prediction(
            self.scenario,
            predictions=predictions
        )
        return predictions
