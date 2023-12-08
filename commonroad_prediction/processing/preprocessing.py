# third-party imports
import numpy as np

# commonroad-io imports
from commonroad.scenario.obstacle import ObstacleRole

# internal imports
from commonroad_prediction.agents.vehicle import Vehicle
from commonroad_prediction.utils import calculate_distance, calculate_vtr_angle
from commonroad_prediction.goal_estimation import GoalEstimation


class Preprocessing:

    @staticmethod
    def get_obstacles_in_view(scenario, ego_id: int, ego_state, fov: float, radius: float):
        """
        Get all the obstacles that are in the field of view of the ego

        Args:
            scenario (Scenario): Considered Scenario.
            ego_id (int): ID of the ego vehicle.
            ego_state (State): State of the ego vehicle.
            fov (float): Considered field of view in °.

        Returns:
            [int]: List with the IDs of the obstacles that can be found within the ego vehicles field of view.
        """
        obstacles_within_view = []
        for obstacle in scenario.obstacles:
            # do not consider the ego vehicle
            if obstacle.obstacle_id != ego_id:
                occ = obstacle.occupancy_at_time(ego_state.time_step)
                # if the obstacle is not in the lanelet network at the given time, its occupancy is None
                if occ is not None:
                    # calculate the distance between the two obstacles
                    dist = calculate_distance(
                        pos1=ego_state.position,
                        pos2=obstacle.occupancy_at_time(
                            ego_state.time_step).shape.center,
                    )
                    # calculate angle between vehicle orientation and obstacle position
                    v_x = round(ego_state.velocity *
                                np.cos(ego_state.orientation + np.pi), 2)
                    v_y = round(ego_state.velocity *
                                np.sin(ego_state.orientation + np.pi), 2)
                    _, angle_deg = calculate_vtr_angle(
                        vtr1=np.array([v_x, v_y]),
                        vtr2=ego_state.position -
                        obstacle.occupancy_at_time(
                            ego_state.time_step).shape.center
                    )
                    # add obstacles that are close enough and in the field of view
                    if abs(angle_deg) < (fov / 2) and dist < radius:
                        obstacles_within_view.append(obstacle.obstacle_id)

        return obstacles_within_view

    @staticmethod
    def get_obstacles_in_radius(scenario, ego_id: int, ego_state, radius: float):
        """
        Get all the obstacles that can be found in a given radius.

        Args:
            scenario (Scenario): Considered Scenario.
            ego_id (int): ID of the ego vehicle.
            ego_state (State): State of the ego vehicle.
            radius (float): Considered radius.

        Returns:
            [int]: List with the IDs of the obstacles that can be found in the ball with the given radius centering at the ego vehicles position.
        """
        obstacles_within_radius = []
        for obstacle in scenario.obstacles:
            # do not consider the ego vehicle
            if obstacle.obstacle_id != ego_id:
                occ = obstacle.occupancy_at_time(ego_state.time_step)
                # if the obstacle is not in the lanelet network at the given time, its occupancy is None
                if occ is not None:
                    # calculate the distance between the two obstacles
                    dist = calculate_distance(
                        pos1=ego_state.position,
                        pos2=obstacle.occupancy_at_time(
                            ego_state.time_step).shape.center,
                    )
                    # add obstacles that are close enough
                    if dist < radius:
                        obstacles_within_radius.append(obstacle.obstacle_id)

        return obstacles_within_radius

    @staticmethod
    def convert_to_prediction_object(scenario, config: dict, obstacle_id: int, timestep: int = 0):

        # get obstacle
        obstacle = scenario.obstacle_by_id(obstacle_id)
        obstacle_state = obstacle.state_at_time(timestep)

        if obstacle.obstacle_role != ObstacleRole.DYNAMIC:
            return None

        # convert to prediction object
        len_traj = len(obstacle.prediction.occupancy_set)
        start_pos = obstacle_state.position
        goal_pos = obstacle.occupancy_at_time(len_traj - 1).shape.center
        pred_obj = Vehicle(start_pos, goal_pos, obstacle_id,
                           initial_velocity=obstacle_state.velocity,
                           orientation=np.degrees(obstacle_state.orientation)
                           )

        # estimate goals by performing a breadth search within the lanelet network
        starting_pos, goals = GoalEstimation.estimate_goals(
            scenario, pred_obj, config)
        pred_obj.predicted_goals = goals
        pred_obj.prediction_starting_position = starting_pos
        return pred_obj
