# ROS imports
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup


class AWInterface(Node):
   def __init__(self):

        # a list to save dynamic obstacles id. key: from cr, value: from autoware
        self.dynamic_obstacles_ids = {}
        self.ego_vehicle_state: State = None
        self.is_predicting = False
        self.scenario = None

        # initialize predictor
        # TODO: get dt and pred_horizon from somewhere else
        self.dt = 0.1
        self.pred_horizon = 3
        pred_horizon_in_seconds = config.prediction.pred_horizon_in_s
        pred_horizon_in_timesteps = int(self.pred_horizon / self.dt)
        self.predictor = PredictionModule(
            scenario, timesteps=pred_horizon_in_timesteps, dt=self.dt)

        self.build_scenario()  # see CR interface, build scenario from osm map
        # create callback group for async execution
        self.callback_group = ReentrantCallbackGroup()

        # vars to save last messages
        self.last_msg_state = None
        self.last_msg_static_obs = None
        self.last_msg_dynamic_obs = None

        # subscribe current position of vehicle
        self.current_state_sub = self.create_subscription(
            Odometry,
            '/localization/kinematic_state',
            self.current_state_callback,
            1,
            callback_group=self.callback_group
        )
        # subscribe static obstacles
        self.static_obs_sub = self.create_subscription(
            DetectedObjects,
            '/perception/object_recognition/detection/objects',
            self.static_obs_callback,
            1,
            callback_group=self.callback_group
        )
        # subscribe dynamic obstacles
        self.dynamic_obs_sub = self.create_subscription(
            PredictedObjects,
            '/perception/object_recognition/objects',
            self.dynamic_obs_callback,
            1,
            callback_group=self.callback_group
        )

        # TODO: publish messages
         # Publishers:
         # /maneuver: visualization_msgs/msg/MarkerArray
         # /objects: autoware_auto_perception_msgs/msg/PredictedObjects
         # /parameter_events: rcl_interfaces/msg/ParameterEvent
         # /rosout: rcl_interfaces/msg/Log

        # TODO: timers needed?
        # create a timer to update scenario
        self.timer_update_scenario = self.create_timer(
            timer_period_sec=0.5,
            callback=self.update_scenario, callback_group=self.callback_group)

        # TODO: create timer to run prediction
        self.timer_prediction = self.create_timer(
            timer_period_sec=0.5,
            callback=self.update_scenario, callback_group=self.callback_group)

    def build_scenario(self):
        """
        Transform map from osm/lanelet2 format to commonroad scenario format.
        """
        map_filename = self.get_parameter(
            'map_osm_file').get_parameter_value().string_value
        left_driving = self.get_parameter(
            'left_driving').get_parameter_value().bool_value
        adjacencies = self.get_parameter(
            'adjacencies').get_parameter_value().bool_value
        self.scenario = lanelet_to_commonroad(map_filename,
                                              proj=self.proj_str,
                                              left_driving=left_driving,
                                              adjacencies=adjacencies)
        if self.write_scenario:
            # save map
            self._write_scenario(self.filename)

    def update_scenario(self):
        """
        Update the commonroad scenario with the latest vehicle state and obstacle messages received.
        """
        # process last state message
        if self.last_msg_state is not None:
            self._process_current_state()
        else:
            self.get_logger().info("has not received a vehicle state yet!")
            return

        # process last static obstacle message
        self._process_static_obs()

        # process last dynamic obstacle message
        self._process_dynamic_obs()

        # plot scenario if plot_scenario = True
        if self.get_parameter('plot_scenario').get_parameter_value().bool_value:
            self._plot_scenario()

        if self.write_scenario:
            self._write_scenario(self.filename)

    def predict(self):

        if not self.is_predicting:
            self.is_predicting = True
            self.predictor.update_scenario(self.scenario)
            predictions = predictor.main_prediction(self.ego_vehicle_state, config.prediction.sensor_radius,
                                                    [float(config.planning.planning_horizon)])
            self.is_predicting = False


# load map
# get autoware tracked_objects state
# get static obstacles
# put them in the map
# perform prediction on them
# publish the result
