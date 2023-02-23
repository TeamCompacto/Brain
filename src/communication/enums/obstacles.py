from enum import Enum

class Obstacles(Enum):

    # Traffic signs

    STOP_SIGN = 1
    PARKING_SIGN = 2
    PRIORITY_SIGN = 3
    CROSSWALK_SIGN = 4
    HIGHWAY_ENTRANCE_SIGN = 5
    HIGHWAY_EXIT_SIGN = 6
    ROUND_ABOUT_SIGN = 7
    ONE_WAY_ROAD_SIGN = 8
    NO_ENTRY_ROAD_SIGN = 9


    # Road marks
    
    DASHED_LANE_MARKS = 10
    CONTINUOUS_LANE_MARKS = 11
    CURVED_LANE_MARKS = 12


    # Other

    TRAFFIC_LIGHTS = 13
    PEDESTRIAN = 14
    OBSTACLE_VEHICLE = 15
    CLOSE_ROAD_STAND = 16
