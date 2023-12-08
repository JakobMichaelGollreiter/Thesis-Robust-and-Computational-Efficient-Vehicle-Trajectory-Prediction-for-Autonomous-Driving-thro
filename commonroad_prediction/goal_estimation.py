# standard imports
from typing import List, Tuple

# third-party imports
import numpy as np

# internal imports
from commonroad_prediction.utils import increase_path_resolution

# globals
LOG_ALL = False
CR_ID = 99999  # 2547 # 2331


class GoalEstimation:

    @staticmethod
    def estimate_goals(scenario, agent, config) -> Tuple[np.array, List[np.array]]:
        """
        Returns projected starting point as well as list of lanelet ids that are potential vehicles routes
        :param scenario: current commonroad scenario
        :param agent: PredictionObject that is to be evaluated
        :return: lanelet ids of potential paths
        """

        # init variables
        lanelet_network = scenario.lanelet_network
        agent_orientation = agent.orientation
        lanelets_in_driving_direction = []
        route_starting_position = []
        goal_positions = []

        # get lanelets that are in close proximity of the agent
        starting_lanelet_candidates = lanelet_network.lanelets_in_proximity(
            agent.position, float(config['PROXIMITY_RADIUS']))

        # get best candidate(s)
        if len(starting_lanelet_candidates) > 0:

            best_candidate = None
            min_deviation = 999
            # get lanelet that fits best
            # TODO: return multiple possitble lanelets ordered by "confidence"
            for lanelet in starting_lanelet_candidates:

                # calculate lanelet distance
                lanelet_center_vertices = increase_path_resolution(
                    lanelet.center_vertices)
                relative_positions = abs(
                    lanelet_center_vertices - agent.position)
                min_distance = min(
                    relative_positions[:, 0] + relative_positions[:, 1])
                projected_agent_position = lanelet_center_vertices[min_distance.argmin(
                )]

                # get lanelet orientation
                try:
                    lanelet_orientation = lanelet.orientation_by_position(
                        agent.position)
                except:
                    continue
                lanelet_orientation = lanelet_orientation / np.pi * 180

                # choose as best candidate if deviation is smaller than previous ones
                deviation = min_distance / float(config['PROXIMITY_RADIUS']) + abs(
                    agent_orientation - lanelet_orientation) * float(config['ORIENTATION_WEIGHT_FACTOR'])
                if deviation < min_deviation:
                    best_candidate = lanelet
                    route_starting_position = projected_agent_position
                    min_deviation = deviation

                if not LOG_ALL and agent.cr_id == CR_ID:
                    print(agent.cr_id, lanelet.lanelet_id, deviation,
                          agent_orientation, lanelet_orientation, min_distance)

            if not LOG_ALL and agent.cr_id == CR_ID:
                print("Choosing ", best_candidate.lanelet_id, lowest_value)
            lanelets_in_driving_direction.append(best_candidate)

            # estimate goal locations based on lanelet successors
            goal_positions = []
            for lanelet in lanelets_in_driving_direction:
                potential_route_candidates = lanelet.find_lanelet_successors_in_range(
                    lanelet_network, max_length=float(config['PATH_SEARCH_DEPTH']))
                for path in potential_route_candidates:
                    last_lanelet = lanelet_network.find_lanelet_by_id(path[-1])
                    last_vertices_center_line = last_lanelet.center_vertices[-1]
                    goal_positions.append(last_vertices_center_line)

        if LOG_ALL or (not LOG_ALL and agent.cr_id == CR_ID):
            print("FINAL", agent.cr_id, agent.position, len(
                lanelets_in_driving_direction), goal_positions)
        return route_starting_position, goal_positions
