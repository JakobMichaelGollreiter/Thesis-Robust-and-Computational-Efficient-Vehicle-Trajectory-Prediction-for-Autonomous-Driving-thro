# standard imports
from typing import Set

# third party
import numpy as np

# internal
from commonroad_prediction.prediction_type import PredictionType


class PredictionObject:

    def __init__(
        self,
        initial_position: Set[float],
        goal_position: Set[float],
        id: int,
        prediction_type: PredictionType,
        orientation: float,
        initial_velocity: float,
        max_velocity: float,
        max_acceleration: float
    ):

        self.cr_id = id
        self.trajectory = list()
        self.reference_trajectory = list()
        self.position = np.array(initial_position)
        self.orientation = orientation
        self.velocity = initial_velocity
        self.goal = np.array(goal_position)
        self.predicted_goals = list()
        self.prediction_trajectories = list()
        self.prediction_starting_position = None
        self.prediction = list()
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.prediction_type = prediction_type

    def get_cartesian_velocities(self):

        v_x = round(self.velocity * np.cos(self.orientation * np.pi / 180.), 2)
        v_y = round(self.velocity * np.sin(self.orientation * np.pi / 180.), 2)
        return {"x": v_x, "y": v_y}
