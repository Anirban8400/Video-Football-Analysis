import sys
sys.path.append('../')
from utils import get_centre_box,measure_distance

class PlayerBallAssigner():
    def  __init__(self):
        self.max_player_ball_dist=70

    def assign_ball_to_player(self, player,ball_bbox):
        ball_position=get_centre_box(ball_bbox)
        min_dist=9999
        assigned_player=-1

        for player_id , player in player.items():
            player_bbox=player['bbox']

            distance_left_foot=measure_distance((player_bbox[0],player_bbox[-1]),ball_position)
            distance_right_foot=measure_distance((player_bbox[2],player_bbox[-1]),ball_position)

            dist=min(distance_right_foot,distance_left_foot)

            if dist<self.max_player_ball_dist:
                if dist<min_dist:
                    min_dist=dist
                    assigned_player=player_id

        return assigned_player
