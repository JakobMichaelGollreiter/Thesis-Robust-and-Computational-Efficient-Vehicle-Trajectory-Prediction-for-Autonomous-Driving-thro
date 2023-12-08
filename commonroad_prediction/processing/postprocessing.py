# third-party imports
import numpy as np


class Postprocessing:

    @staticmethod
    def add_orientation_velocity_and_shape_to_prediction(
        scenario, predictions: dict, safety_margin_length=1.0, safety_margin_width=0.5
    ):
        """
        Extend the prediction by adding information about the orientation, velocity and the shape of the predicted obstacle.

        Args:
            predictions (dict): Prediction dictionary that should be extended.

        Returns:
            dict: Extended prediction dictionary.
        """
        # go through every predicted obstacle
        obstacle_ids = list(predictions.keys())
        for obstacle_id in obstacle_ids:
            obstacle = scenario.obstacle_by_id(obstacle_id)
            # get x- and y-position of the predicted trajectory
            pred_traj = predictions[obstacle_id]['pos_list']
            pred_length = len(pred_traj)

            # there may be some predictions without any trajectory (when the obstacle disappears due to exceeding time)
            if pred_length == 0:
                del predictions[obstacle_id]
                continue

            # for predictions with only one timestep, the gradient can not be derived --> use initial orientation
            if pred_length == 1:
                pred_orientation = [obstacle.initial_state.orientation]
                pred_v = [obstacle.initial_state.velocity]
            else:
                t = [0.0 + i * scenario.dt for i in range(pred_length)]
                x = pred_traj[:, 0][0:pred_length]
                y = pred_traj[:, 1][0:pred_length]

                # calculate the yaw angle for the predicted trajectory
                dx = np.gradient(x, t)
                dy = np.gradient(y, t)
                # if the vehicle does barely move, use the initial orientation
                # otherwise small uncertainties in the position can lead to great orientation uncertainties
                if all(dxi < 0.0001 for dxi in dx) and all(dyi < 0.0001 for dyi in dy):
                    init_orientation = obstacle.initial_state.orientation
                    pred_orientation = np.full(
                        (1, pred_length), init_orientation)[0]
                # if the vehicle moves, calculate the orientation
                else:
                    pred_orientation = np.arctan2(dy, dx)

                # get the velocity from the derivation of the position
                pred_v = np.sqrt((np.power(dx, 2) + np.power(dy, 2)))

            # add the new information to the prediction dictionary
            predictions[obstacle_id]['orientation_list'] = pred_orientation
            predictions[obstacle_id]['v_list'] = pred_v
            obstacle_shape = obstacle.obstacle_shape
            predictions[obstacle_id]['shape'] = {
                'length': obstacle_shape.length + safety_margin_length,
                'width': obstacle_shape.width + safety_margin_width,
            }

        # return the updated predictions dictionary
        return predictions
