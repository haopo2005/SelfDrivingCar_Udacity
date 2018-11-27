from collections import namedtuple
from math import sqrt, exp

TrajectoryData = namedtuple("TrajectoryData", [
    'intended_lane',
    'final_lane',
    'end_distance_to_goal',
    ])

"""
Here we have provided two possible suggestions for cost functions, but feel free to use your own!
The weighted cost over all cost functions is computed in calculate_cost. See get_helper_data
for details on how useful helper data is computed.
"""

#weights for costs
REACH_GOAL = 0.9
EFFICIENCY = 0.1

DEBUG = False


def goal_distance_cost(vehicle, trajectory, predictions, data):
    """
    Cost increases based on distance of intended lane (for planning a lane change) and final lane of a trajectory.
    Cost of being out of goal lane also becomes larger as vehicle approaches goal distance.
    """
    delta_d = 2.0*vehicle.goal_lane - data[0] - data[1]
    cost = 1 - exp(-(abs(delta_d) / data[2]))
    return cost


def inefficiency_cost(vehicle, trajectory, predictions, data):
    """
    Cost becomes higher for trajectories with intended lane and final lane that have slower traffic. 
    """
    #If no vehicle is in the proposed lane, we can travel at target speed.
    speed_intended = velocity(predictions, data[0]) or vehicle.target_speed # vehicle.target_speed is really confused!! 假设该车道没有汽车或该车道不存在，则以最大速度运行？
    speed_final = velocity(predictions, data[1]) or vehicle.target_speed # vehicle.target_speed is really confused!!
    #print(data[0], data[1])
    #print(speed_final,speed_intended, vehicle.target_speed)
    cost = (2.0*vehicle.target_speed - speed_intended - speed_final)/vehicle.target_speed
    return cost


def calculate_cost(vehicle, trajectory, predictions):
    """
    Sum weighted cost functions to get total cost for trajectory.
    """
    trajectory_data = get_helper_data(vehicle, trajectory, predictions)
    #print( vehicle.lane, vehicle.s, vehicle.v, vehicle.a, vehicle.state)
    cost = 0.0
    cf_list = [goal_distance_cost, inefficiency_cost]
    weight_list = [REACH_GOAL, EFFICIENCY]

    for weight, cf in zip(weight_list, cf_list):
        new_cost = weight*cf(vehicle, trajectory, predictions, trajectory_data)
        if DEBUG:
            print ("{} has cost {} for lane {}".format(cf.__name__, new_cost, trajectory[-1].lane))
        cost += new_cost
    return cost

def get_helper_data(vehicle, trajectory, predictions):
    """
    Generate helper data to use in cost functions:
    indended_lane:  +/- 1 from the current lane if the ehicle is planning or executing a lane change.
    final_lane: The lane of the vehicle at the end of the trajectory. The lane is unchanged for KL and PLCL/PLCR trajectories.
    distance_to_goal: The s distance of the vehicle to the goal.

    Note that indended_lane and final_lane are both included to help differentiate between planning and executing
    a lane change in the cost functions.
    """

    last = trajectory[1]

    if last.state == "PLCL":
        intended_lane = last.lane + 1
    elif last.state == "PLCR":
        intended_lane = last.lane - 1
    else:
        intended_lane = last.lane

    distance_to_goal = vehicle.goal_s - last.s
    final_lane = last.lane

    return TrajectoryData(
        intended_lane,
        final_lane,
        distance_to_goal)


def velocity(predictions, lane):
    """
    All non ego vehicles in a lane have the same speed, so to get the speed limit for a lane,
    we can just find one vehicle in that lane.
    """
    for v_id, predicted_traj in predictions.items():
        if predicted_traj[0].lane == lane and v_id != -1:
            return predicted_traj[0].v
