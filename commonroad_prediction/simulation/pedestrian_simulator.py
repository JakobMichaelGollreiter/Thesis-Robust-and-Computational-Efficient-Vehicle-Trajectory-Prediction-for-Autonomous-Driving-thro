#!/usr/bin/env python3

# standard imports
from copy import deepcopy
from typing import Union
import math

# third-party imports
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as lines

# commonroad imports
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.prediction.prediction import TrajectoryPrediction, Trajectory, Occupancy
from commonroad.geometry.shape import Circle, Rectangle

# internal importsMou
# TODO: ModuleNotFoundError
from pedestrian_behavior_simulation.python_implementation.objects.pedestrian import Pedestrian
from pedestrian_behavior_simulation.python_implementation.objects.static_obstacle import StaticObstacle
from pedestrian_behavior_simulation.python_implementation.behavior_models.gradient_based_model_flo import GradientBasedModel
from pedestrian_behavior_simulation import parameters



class _ModelPrediction(TrajectoryPrediction):
    """Class used to dynamically overload the CommonRoad `TrajectoryPrediction` class with custom behavior."""

    def init_from_trajectory_prediction(self, scenario: Scenario):
        self.scenario = scenario

        self.time_steps = len(self.occupancy_set)
        self.prev_time_step = 0
        self.prev_occupancy = deepcopy(self.occupancy_set[0])

        self.simulation = GradientBasedModel()
        self.simulation.show = False

        # Pedestrian
        self.simulation.add_pedestrian(Pedestrian(
            (self.prev_occupancy.shape.center[0],
             self.prev_occupancy.shape.center[1]),
            (78.0, 45.6), initial_velocity=1.0, comf_velocity=1.0, max_velocity=1, orientation=0, body_radius=0.1, id=0))

        self.car_1 = None
        self.car_2 = None
        for obs in scenario.dynamic_obstacles:
            if obs.prediction._occupancy_set == self._occupancy_set:
                continue
            if self.car_1 is None:
                self.car_1 = obs
            elif self.car_2 is None:
                self.car_2 = obs
                break
        assert self.car_1 is not None
        assert self.car_2 is not None

        # Car #1
        car_1_occ = self.car_1.prediction._occupancy_set
        self.car_1_goal = (car_1_occ[len(car_1_occ) - 1]._shape.center[0],
                           car_1_occ[len(car_1_occ) - 1]._shape.center[1])
        self.simulation.add_pedestrian(Pedestrian(
            (car_1_occ[0]._shape.center[0], car_1_occ[0]._shape.center[1]),
            self.car_1_goal,
            initial_velocity=10, comf_velocity=10, max_velocity=10, orientation=90, id=1))
        # Car #2
        car_2_occ = self.car_2.prediction._occupancy_set
        self.car_2_goal = (car_2_occ[len(car_2_occ) - 1]._shape.center[0],
                           car_2_occ[len(car_2_occ) - 1]._shape.center[1])
        self.simulation.add_pedestrian(Pedestrian(
            (car_2_occ[0]._shape.center[0], car_2_occ[0]._shape.center[1]),
            self.car_2_goal,
            initial_velocity=10, comf_velocity=10, max_velocity=10, orientation=90, id=2))

        # Pedestrian Walk Way
        self.simulation.add_static_obstacle(StaticObstacle(
            (40, 47.2),
            (80, 47.2),
        ))
        self.simulation.add_static_obstacle(StaticObstacle(
            (40, 44.1),
            (80, 44.1),
        ))

    def occupancy_at_time_step(self, time_step: int) -> Union[None, Occupancy]:
        """Basic example of a dynamically calculated model that uses its environment."""
        # If out of range or initial.
        if self.time_steps == 0 or time_step > self.time_steps:
            return None

        for i in range(0, int((time_step - self.prev_time_step) / parameters.GLOBAL.DT)):
            # Reset car #1
            car_1_occ = self.car_1.prediction._occupancy_set[time_step]._shape.center
            self.simulation.objects["pedestrian"][1] = Pedestrian(
                (car_1_occ[0], car_1_occ[1]), self.car_1_goal,
                initial_velocity=16, comf_velocity=16, orientation=90, id=1)
            # Reset car #2
            car_2_occ = self.car_2.prediction._occupancy_set[time_step]._shape.center
            self.simulation.objects["pedestrian"][2] = Pedestrian(
                (car_2_occ[0], car_2_occ[1]), self.car_2_goal,
                initial_velocity=5, comf_velocity=5, orientation=90, id=2)

            self.simulation.run(timesteps=1)

        self.prev_time_step = time_step
        self.prev_occupancy.shape.center[0] = self.simulation.objects["pedestrian"][0].position[0]
        self.prev_occupancy.shape.center[1] = self.simulation.objects["pedestrian"][0].position[1]

        if True:
            fig = plt.gcf()
            ax = fig.gca()
            # ax.add_line(lines.Line2D((40, 80),
            #                          (47.2, 47.2), color="r", zorder=1000))
            # ax.add_line(lines.Line2D((40, 80),
            #                          (44.1, 44.1), color="r", zorder=1000))

        if True:
            ped = plt.Circle(
                (self.simulation.objects["pedestrian"][0].position[0],
                    self.simulation.objects["pedestrian"][0].position[1]), 0.5, color='r', zorder=1000)
            car_1 = plt.Circle(
                (self.simulation.objects["pedestrian"][1].position[0],
                    self.simulation.objects["pedestrian"][1].position[1]), 0.5, color='g', zorder=1000)
            car_2 = plt.Circle(
                (self.simulation.objects["pedestrian"][2].position[0],
                    self.simulation.objects["pedestrian"][2].position[1]), 0.5, color='b', zorder=10)

            fig = plt.gcf()
            ax = fig.gca()

            ax.add_patch(car_1)
            ax.add_patch(car_2)
            ax.add_patch(ped)

        return self.prev_occupancy


class SimulationContext:
    """Class unifying all of the different model instances in a commonroad scenario."""

    def __init__(self, scenario: Scenario):
        self.scenario = scenario

    def overload_dynamic_obstacle(self, obstacle_id: int):
        """Overloads the dynamic obstacle with simulated behavior."""
        pedestrian = None
        for obstacle in self.scenario.dynamic_obstacles:
            if obstacle.obstacle_id == obstacle_id:
                pedestrian = obstacle
        assert pedestrian is not None

        pedestrian.prediction.__class__ = _ModelPrediction
        pedestrian.prediction.init_from_trajectory_prediction(self.scenario)
