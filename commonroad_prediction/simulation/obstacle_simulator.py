# third-party imports
import numpy as np

# commonroad imports
from commonroad.scenario import Scenario
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.prediction.prediction import TrajectoryPrediction, Trajectory
from commonroad.geometry.shape import Circle, Rectangle


class ObstacleSimulator:

    def __init__(self, scenario: Scenario, delta_t: float, init_counter: int = 0):
        self.id_counter = init_counter
        self.simulation_scenario = scenario
        self.dt = delta_t
        self.obstacle_list = list()
        self.next_timestep = 1
        self.prediction_module = PredictionModule(
            PREDICTION_TIMESTEPS, self.dt)

    def _get_id(self):
        unique_id = self.id_counter
        self.id_counter += 1
        return unique_id

    def spawn(self, obstacle: PredictionObject, init_time_step: int = 0):

        # create necessary commonroad objects before defining obstacle
        current_state = Trajectory(init_time_step, obstacle.trajectory)
        if obstacle.type == "car":
            obstacle_shape = Rectangle(
                obstacle.dimensions[1], obstacle.dimensions[0])
        else:
            obstacle_shape = Circle(obstacle.radius)
        obstacle_movement = TrajectoryPrediction(current_state, obstacle_shape)

        # create dynamic commonroad obstacle
        obstacle.cr_id = self._get_id()
        cr_object = DynamicObstacle(
            obstacle_id=obstacle.cr_id,
            obstacle_type=ObstacleType(obstacle.type),
            obstacle_shape=obstacle_shape,
            initial_state=State(
                position=obstacle.position,
                orientation=np.radians(obstacle.orientation),
                velocity=obstacle.velocity,
                time_step=init_time_step
            ),
            prediction=obstacle_movement
        )

        # add to commonroad scenario and obstacle list
        self.simulation_scenario.add_objects(cr_object)
        self.obstacle_list.append((obstacle, cr_object))

    def simulate_timestep(self, init_time_step: int = 0):
        for obstacle, cr_ref in self.obstacle_list:
            # update velocity
            obstacle.velocity = min(
                obstacle.velocity + obstacle.max_acceleration,
                obstacle.max_velocity
            )

            # update position
            current_state = obstacle.reference_trajectory[self.next_timestep]
            obstacle.position = current_state.position
            obstacle.trajectory.append(current_state)
            obstacle.orientation = np.rad2deg(current_state.orientation)

            # predict trajectories
            for obstacle, _ in self.obstacle_list:
                self.prediction_module.predict_trajectory(
                    obstacle, self.next_timestep + 1)

            # update commonroad state
            updated_trajectory = Trajectory(
                init_time_step, obstacle.trajectory + obstacle.prediction['state_list'])
            if obstacle.type == "car":
                updated_movement = TrajectoryPrediction(updated_trajectory, Rectangle(
                    obstacle.dimensions[1], obstacle.dimensions[0]))
            else:
                updated_movement = TrajectoryPrediction(
                    updated_trajectory, Circle(obstacle.radius))
            cr_ref.prediction = updated_movement

        # update timestep for next run
        self.next_timestep += 1
