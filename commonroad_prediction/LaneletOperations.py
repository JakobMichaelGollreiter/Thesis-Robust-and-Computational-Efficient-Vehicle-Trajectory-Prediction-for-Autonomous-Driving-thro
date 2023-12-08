import commonroad_prediction.HelperFunctions as HelperFunctions
import math
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet

def fillAllPossibleLanes(flat_list_lanes, egoArray, ix, iy):
    if not flat_list_lanes:
        return egoArray
    else:
        egoArray[ix, iy] = 0.5
    return egoArray


def fillAllReachableLanes(flat_list_lanes, flat_list_merged_lanelets, curLanelet, egoArray, ix, iy):
    if flat_list_lanes and flat_list_merged_lanelets:
        overlap = list(set(flat_list_lanes) &
                        set(flat_list_merged_lanelets))
        laneletID = -1 if curLanelet is None else curLanelet.lanelet_id
        if overlap or laneletID in flat_list_lanes:
            egoArray[ix, iy] = 1
            return egoArray
    return egoArray

def merge_all_merged_lanelets_in_proximity(lanelets_in_proximity, merged_lanelets, position, orientation_of_vehicle, LaneletNetwork):
    for lanelet in lanelets_in_proximity:
        lanelet_orientation = lanelet.orientation_by_position_without_assertion(position)
        orientation_difference = abs(HelperFunctions.angle_difference(lanelet_orientation, orientation_of_vehicle))
        if orientation_difference > math.pi/6:
            continue
        merged_lanelets_from_one_lane = Lanelet.all_lanelets_by_merging_successors_from_lanelet(
            lanelet, LaneletNetwork, max_length=100.0)
        merged_lanelets.extend(merged_lanelets_from_one_lane[1])
    return merged_lanelets

def get_closest_lanelet(point_list, LaneletNetwork):
    idsOfLanelet = LaneletNetwork.find_lanelet_by_position(point_list) # ToDo handle edge case if no lanelet is found
    if len(idsOfLanelet) > 0:
        closest_lanelet = idsOfLanelet[0][0]
    else:
        closest_lanelet = None  # or assign a default value
        return None
    return LaneletNetwork.find_lanelet_by_id(closest_lanelet)

def get_all_lanelets_with_different_successor_options(position, orientation_of_vehicle, LaneletNetwork):
    lanelets_in_proximity = LaneletNetwork.lanelets_in_proximity(position, 4.0) # get lanelets in proximity from current position # the float is to set the "proximity_radius"
    try:
        closest_lanelet = get_closest_lanelet([position], LaneletNetwork)  # get closest_lanelet from current position
    except Exception:
        print('Error: No closest lanelet found... Now trying: LaneletNetwork.lanelets_in_proximity(position, 40.0)[0]')
        closest_lanelet = LaneletNetwork.lanelets_in_proximity(position, 40.0)[0]  # ToDo Try without closest lanelet

    merged_lanelets = []  # merged_lanelets merge all lanelet successors
    merged_lanelets.extend(Lanelet.all_lanelets_by_merging_successors_from_lanelet(closest_lanelet, LaneletNetwork, max_length=100.0)[1])  # ToDo Try without closest lanelet
    merged_lanelets = merge_all_merged_lanelets_in_proximity(lanelets_in_proximity, merged_lanelets, position=position, orientation_of_vehicle=orientation_of_vehicle, LaneletNetwork=LaneletNetwork)
    return merged_lanelets, closest_lanelet