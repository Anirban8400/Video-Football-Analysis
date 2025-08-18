from utils import read_video, save_video
from tracking import Tracker
import cv2
import os
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from perspective_change import ViewTransformer
from speed_dist_estimator import SpeedAndDistance_Estimator

def main():
    video_frames=read_video(r'D:\Football_Project - Copy\Data\08fd33_4.mp4')
    tracker=Tracker(r'D:\Football_Project\models\best.pt')
    tracks=tracker.get_object_tracks(video_frames,read_from_stubs=True, stub_path='stubs/track_stubs.pkl')

    #extimate camera movements
    camera_movement_estimator=CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame=camera_movement_estimator.get_camera_movement(video_frames,
                                                                            read_from_stub=True,
                                                                            stub_path='stubs/camera_movement.pkl')
    
     # View Trasnformer
    perspectiveChange = ViewTransformer()
    perspectiveChange.add_transformed_position_to_tracks(tracks)

    #interpolation
    tracks["ball"]=tracker.interpolate_ball_position(tracks["ball"])

    # Speed and distance estimator
    speed_dist_estimator = SpeedAndDistance_Estimator()
    speed_dist_estimator.add_speed_and_distance_to_tracks(tracks)


    #save cropped
    for track_id,player in tracks['player'][0].items():
        bbox=player["bbox"]
        frame=video_frames[0]
        #crop players from frame
        cropped_image=frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        #save image
        cv2.imwrite(f'output_final/cropped_img.jpg', cropped_image)
        break
    team_assigner=TeamAssigner()

    team_assigner.assign_team_color(video_frames[0], tracks['player'][0])
    for frame_num , player_track in enumerate(tracks['player']):
        for player_id, track in player_track.items():
            team=team_assigner.get_player_team(video_frames[frame_num],
                                               track['bbox'],
                                               player_id)
            tracks['player'][frame_num][player_id]['team']=team
            tracks['player'][frame_num][player_id]['team_color'] = tuple(map(int, team_assigner.team_colors[team]))


    #Assign ball to player

    player_assigner=PlayerBallAssigner()
    team_ball_control=[]

    for frame_num , player_track in enumerate(tracks['player']):
        ball_bbox=tracks['ball'][frame_num][0]['bbox']
        assigned_player=player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player!=-1:
            tracks['player'][frame_num][assigned_player]['has_ball']=True
            team_ball_control.append(tracks['player'][frame_num][assigned_player]['team'])

        else:
            if team_ball_control:   # not empty
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(1) 
    team_ball_control=np.array(team_ball_control)

        
                                               


    os.makedirs('output_final', exist_ok=True)
    output_video_frames=tracker.draw_annotations(video_frames,tracks, team_ball_control)

    #draw cam movement
    output_video_frames=camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    ## Draw Speed and Distance
    speed_dist_estimator.draw_speed_and_distance(output_video_frames,tracks)

    save_video(output_video_frames,'output_final/output_video.mp4' )

if __name__=='__main__':
    main()