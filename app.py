# app.py
import streamlit as st
import tempfile
import os
import numpy as np
import cv2

# import your modules
from utils import read_video, save_video
from tracking import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from perspective_change import ViewTransformer
from speed_dist_estimator import SpeedAndDistance_Estimator


def process_video(input_path, output_path):
    """Runs your full football analysis pipeline on a video."""
    video_frames = read_video(input_path)
    tracker = Tracker(r'D:\Football_Project\models\best.pt')

    # object tracks
    tracks = tracker.get_object_tracks(
        video_frames, read_from_stubs=True, stub_path='stubs/track_stubs.pkl'
    )

    # camera movement
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames, read_from_stub=True, stub_path='stubs/camera_movement.pkl'
    )

    # view transform
    perspectiveChange = ViewTransformer()
    perspectiveChange.add_transformed_position_to_tracks(tracks)

    # interpolate ball
    tracks["ball"] = tracker.interpolate_ball_position(tracks["ball"])

    # speed & distance
    speed_dist_estimator = SpeedAndDistance_Estimator()
    speed_dist_estimator.add_speed_and_distance_to_tracks(tracks)

    # team assign
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['player'][0])

    for frame_num, player_track in enumerate(tracks['player']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['player'][frame_num][player_id]['team'] = team
            tracks['player'][frame_num][player_id]['team_color'] = tuple(
                map(int, team_assigner.team_colors[team])
            )

    # ball possession
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['player']):
        ball_bbox = tracks['ball'][frame_num][0]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['player'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['player'][frame_num][assigned_player]['team'])
        else:
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(1)

    team_ball_control = np.array(team_ball_control)

    # drawing
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame
    )
    speed_dist_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # save processed video
    save_video(output_video_frames, output_path)


def main():
    st.title("⚽ Football Analytics Demo")

    uploaded_file = st.file_uploader("Upload a match video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        # write uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
            tmp_input.write(uploaded_file.read())
            input_path = tmp_input.name

        output_path = os.path.join(tempfile.gettempdir(), "processed_output.mp4")

        with st.spinner("Processing video... please wait."):
            process_video(input_path, output_path)

        st.success("✅ Processing complete!")

        # show processed video in the app
        with open(output_path, "rb") as f:
            video_bytes = f.read()
            st.video(video_bytes, format="video/mp4")

        # download button
        with open(output_path, "rb") as f:
            st.download_button("Download Processed Video", f, file_name="output_video.mp4")


if __name__ == "__main__":
    main()
