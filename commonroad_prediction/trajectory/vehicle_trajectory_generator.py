# standard imports
from typing import List

# third-party imports
import numpy as np

# commonroad-io imports
from commonroad.common.util import Interval
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.state import InitialState
from commonroad.geometry.shape import Circle

# commonroad-route-planner imports
from commonroad_route_planner.route_planner import RoutePlanner

# internal imports
from commonroad_prediction.utils import calculate_vtr_angle, calculate_distance
from commonroad_prediction.agents.prediction_object import PredictionObject


class TrajectoryGenerator:

    @staticmethod
    def _calculate_reference_path(object: PredictionObject, planning_scenario: Scenario, goal: np.array, dt: int):

        # init problem to be solved
        starting_state_on_lanelet = InitialState(
            position=object.prediction_starting_position,
            orientation=object.orientation,
            velocity=object.velocity,
            yaw_rate=0.0,
            slip_angle=0.0,
            time_step=0
        )
        reference_path = [object.reference_trajectory[0]]
        goal_state = InitialState(position=Circle(
            radius=0.5, center=goal), time_step=Interval(0, 100))
        planning_problem = PlanningProblem(
            planning_problem_id=1000,
            initial_state=starting_state_on_lanelet,
            goal_region=GoalRegion([goal_state])
        )
        route_planner = RoutePlanner(planning_scenario, planning_problem)

        # calculate initial reference path
        potential_routes = route_planner.plan_routes()
        try:
            ref_path = potential_routes.retrieve_first_route().reference_path
        except:
            print(
                "Warning: No route could be found. Check if goal is outside of the map.")
            ref_path = [object.position for _ in range(200)]
        vel_dict = object.get_cartesian_velocities()
        vtr_vel_ego = [vel_dict["x"], vel_dict["y"]]

        # convert to potential prediction path based on reference
        budget_until_next_location = object.velocity * dt
        current_time_step = 1
        next_simulation_location = None

        # the first points in the reference path are not alinged with the heading direction
        REFERENCE_PATH_ALGINED_WITH_HEADING = False
        while len(ref_path) > 0 and not REFERENCE_PATH_ALGINED_WITH_HEADING:

            current_path_point, ref_path = ref_path[0], ref_path[1:]
            vtr_relative_position = [
                current_path_point[0] - object.position[0], current_path_point[1] - object.position[1]]
            _, angle_deg = calculate_vtr_angle(
                vtr_vel_ego, vtr_relative_position)

            if abs(angle_deg) < 90:
                REFERENCE_PATH_ALGINED_WITH_HEADING = True

        while len(ref_path) > 0:

            current_path_point, ref_path = ref_path[0], ref_path[1:]
            if isinstance(next_simulation_location, type(None)):
                next_simulation_location = current_path_point
                continue

            distance = calculate_distance(
                current_path_point, next_simulation_location)
            if distance < budget_until_next_location:
                budget_until_next_location -= distance
                next_simulation_location = current_path_point
            else:
                budget_until_next_location += (object.velocity * dt)
                vtr_x_axis = np.array([1, 0])
                vtr_new_orientation = current_path_point - next_simulation_location
                new_orientation = np.arctan2(
                    vtr_new_orientation[1], vtr_new_orientation[0])
                reference_path.append(InitialState(
                    position=np.array(next_simulation_location),
                    orientation=new_orientation,
                    velocity=object.velocity,
                    yaw_rate=0.0,
                    slip_angle=0.0,
                    time_step=current_time_step
                ))
                current_time_step += 1
                next_simulation_location = None

        return reference_path

    @staticmethod
    def _merge_to_vehicle_position(reference_trajectory: List[np.array], config: dict, timesteps: int):

        for t in range(timesteps):

            # get orientation from reference trajectory
            p0 = reference_trajectory[t].position
            p1 = reference_trajectory[t+1].position
            p2 = reference_trajectory[t+2].position
            orientation = [p2[0] - p1[0], p2[1] - p1[1]]

            # correct orientation by adjusting it towards the predicted reference path
            p1_new = p0 + orientation
            error = np.array([(p1_new[0] - p1[0]), (p1_new[1] - p1[1])]
                             ) * float(config['POSITION_CORRECTION_FACTOR'])
            p1_new = p1_new - error

            # set to new position so it can be used as reference in the next step
            reference_trajectory[t+1].position = np.array(p1_new)

    @staticmethod
    def _adjust_for_vehicle_dynamics(object: PredictionObject, reference_trajectory: List[np.array], config: dict, timesteps: int, dt: int, dx: float):

        # get vehicle state
        vel_dict = object.get_cartesian_velocities()
        rel_vel = np.array([vel_dict["x"], vel_dict["y"]]) * dt

        for t in range(timesteps):

            # calculate distance between constant velocity model and reference path
            position_cv = reference_trajectory[t].position + rel_vel
            position_ref = reference_trajectory[t+1].position
            distance = calculate_distance(position_cv, position_ref)

            # check if maximum yaw angle is exceeded
            if distance > dx:
                # correct trajectory based on allowed deviation dx
                pos_difference = np.array(
                    [- position_cv[0] + position_ref[0], - position_cv[1] + position_ref[1]])
                correction = pos_difference * \
                    (dx/distance) * \
                    float(config['DYNAMICS_COMPENSATION_FACTOR'])
                position_adjusted = position_cv + correction

                # update trajectory and relative velocity
                reference_trajectory[t +
                                     1].position = np.array(position_adjusted)
                rel_vel = reference_trajectory[t+1].position - \
                    reference_trajectory[t].position
