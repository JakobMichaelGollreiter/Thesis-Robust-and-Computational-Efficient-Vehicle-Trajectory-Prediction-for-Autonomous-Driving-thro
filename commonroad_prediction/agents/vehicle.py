# standard imports
import time
from typing import List, Set

# third party
import numpy as np

# commonroad-io
from commonroad.common.util import Interval
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.state import InitialState
from commonroad.geometry.shape import Circle

# commonroad-route-planner
from commonroad_route_planner.route_planner import RoutePlanner

# internal
from commonroad_prediction.prediction_type import PredictionType
from commonroad_prediction.utils import calculate_vtr_angle, calculate_distance
from commonroad_prediction.agents.prediction_object import PredictionObject
from commonroad_prediction.trajectory.vehicle_trajectory_generator import TrajectoryGenerator


class Vehicle(PredictionObject):

    def __init__(
        self,
        initial_position: Set[float],
        goal_position: Set[float],
        id: int = None,
        prediction_type: PredictionType = PredictionType("Lanebased"),
        dimensions: Set[float] = (1.85, 4.5),
        orientation: float = 0.0,  # orientation in degrees
        initial_velocity: float = 0.0,
        max_velocity: float = 16.67,
        max_acceleration: float = 3.0
    ):

        super(Vehicle, self).__init__(
            initial_position,
            goal_position,
            id,
            prediction_type,
            orientation,
            initial_velocity,
            max_velocity,
            max_acceleration
        )
        self.dimensions = np.array(dimensions)
        self.type = "car"

        initial_state = InitialState(
            position=self.position,
            orientation=self.orientation,
            velocity=self.velocity,
            yaw_rate=0.0,
            slip_angle=0.0,
            time_step=0
        )
        self.trajectory.append(initial_state)
        self.reference_trajectory.append(initial_state)

    def calculate_prediction_trajectories(self, planning_scenario: Scenario, config: dict, dt: int):

        prediction_reference_trajectories = []
        for goal in self.predicted_goals:
            ref_traj = TrajectoryGenerator._calculate_reference_path(
                self, planning_scenario, goal, dt
            )
            prediction_reference_trajectories.append(ref_traj)

        for reference_trajectory in prediction_reference_trajectories:

            # check if the reference trajectory is long enough for a prediction adjustment
            timesteps = int(
                int(config['PREDICTION_ADJUSTMENT_TIMEFRAME']) / dt)
            minimum_number_of_timesteps = 3
            if len(reference_trajectory) < minimum_number_of_timesteps:
                return
            elif len(reference_trajectory) <= (timesteps + 1):
                timesteps = len(reference_trajectory) - 2

            # check if an adjustment is needed since prediction and initial
            # vehicle position vary too much
            distance = calculate_distance(
                reference_trajectory[1].position, self.position)
            if distance > float(config['LANE_CENTER_THRESHOLD']):

                TrajectoryGenerator._merge_to_vehicle_position(
                    reference_trajectory, config, timesteps)

                if self.velocity == 0:
                    return

                dx = self.velocity * dt * \
                    np.tan((int(config['YAW_RATE'])/180.) * np.pi)
                TrajectoryGenerator._adjust_for_vehicle_dynamics(
                    self, reference_trajectory, config, timesteps, dt, dx)

        self.prediction_trajectories = prediction_reference_trajectories
