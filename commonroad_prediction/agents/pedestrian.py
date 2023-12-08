# standard imports
from typing import Set

# commonroad-io imports
from commonroad.scenario.trajectory import State

# internal imports
from commonroad_prediction.prediction_type import PredictionType
from commonroad_prediction.agents.prediction_object import PredictionObject

# TODO: Add reference path


class Pedestrian(PredictionObject):

    def __init__(
        self,
        initial_position: Set[float],
        goal_position: Set[float],
        prediction_type: PredictionType = PredictionType("Linear"),
        body_radius: float = 0.5,
        orientation: float = 0.0,
        initial_velocity: float = 0,
        max_velocity: float = 1.2,
        max_acceleration: float = 0.5
    ):

        super(Pedestrian, self).__init__(
            initial_position,
            goal_position,
            prediction_type,
            orientation,
            initial_velocity,
            max_velocity,
            max_acceleration
        )
        self.radius = body_radius
        self.type = "pedestrian"

        initial_state = State(
            position=self.position,
            orientation=np.radians(self.orientation),
            velocity=self.velocity,
            time_step=0
        )
        self.trajectory.append(initial_state)
        self.reference_trajectory.append(initial_state)
