from ultralytics import YOLO
import supervision as sv
import pickle
import numpy as np
import pandas as pd
import os
import cv2
import sys
sys.path.append('../') 

"""you’re adding the parent directory of the current script to that search list.
'../' is a relative path meaning “go one folder up.”"""

from utils import get_centre_box, get_width_box


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    
    def interpolate_ball_position(self, ball_positions):
        ball_positions=[x.get(0,{}).get('bbox',[]) for x in ball_positions]   #in  no value then register it as empty
        df_ball_positions=pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])

        #interpolate missing values
        df_ball_positions=df_ball_positions.interpolate()
        df_ball_positions=df_ball_positions.bfill()

        ball_positions=[{0:{"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
        

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.2)
            detections += detections_batch
        return detections
    

    def get_object_tracks(self, frames, read_from_stubs=False, stub_path=None):
        # #Load from stub if requested
        # if read_from_stubs and stub_path and os.path.exists(stub_path):
        #     print("EXTRACTING PICKLE")
        #     with open(stub_path, 'rb') as f:
        #         return pickle.load(f)

        detections = self.detect_frames(frames)

        tracks = {
            "player": [],
            "referee": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert goalkeeper → player
            for obj_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[obj_ind] = cls_names_inv["player"]

            # Track players/referees
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["player"].append({})
            tracks["referee"].append({})
            tracks["ball"].append({})

            for det in detection_with_tracks:
                bbox = det[0].tolist()
                cls_id = det[3]
                track_id = det[4]

                if cls_names[cls_id] == 'player':
                    tracks["player"][frame_num][track_id] = {"bbox": bbox}
                elif cls_names[cls_id] == 'referee':
                    tracks["referee"][frame_num][track_id] = {"bbox": bbox}

            # Detect ball (not tracked)
            for det in detection_supervision:
                bbox = det[0].tolist()
                cls_id = det[3]
                if cls_names[cls_id] in ('ball', 'football'):  # Support both names
                    tracks["ball"][frame_num][0] = {"bbox": bbox}  # Always key 0 for the ball

        # #Save to stub if path given
        # if stub_path:
        #     print("MAKING PICKLE FILE")
        #     with open(stub_path, 'wb') as f:
        #         pickle.dump(tracks, f)

        return tracks

    #draws a traingle on top of the ball/person controlling the ball
    def draw_triangle(self,frame, bbox,color):
        y=int(bbox[1])
        x,_=get_centre_box(bbox)
        tri_p=np.array([[x,y],[x+10,y-20],[x-10,y-20]])
        cv2.drawContours(frame, [tri_p],0,color,cv2.FILLED)
        cv2.drawContours(frame, [tri_p],0,(0,0,0),2)
        return frame

    ##draw a ellipse at the bottom of each player
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2=int(bbox[3])  #we need the bottom y and the ellipse will be there

        x_centre, y_centre=get_centre_box(bbox)
        width=get_width_box(bbox)

        cv2.ellipse(
            frame, 
            center=(x_centre,y2),
            axes=(int(width),int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width=40
        rectangle_height=20
        x2_rec=x_centre+rectangle_width//2
        x1_rec=x_centre-rectangle_width//2
        y2_rec=(y2+rectangle_height//2)+15
        y1_rec=(y2-rectangle_height//2)+15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rec),int(y1_rec)),
                          (int(x2_rec),int(y2_rec)),
                          color,
                          cv2.FILLED)
            x1_text=x1_rec+12
            if track_id>99:   #in case it is  a bigger nuumber
                x1_text-=10
            cv2.putText(frame, f"{track_id}",(int(x1_text),int(y1_rec+15)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

        return frame
    
    #draw a transparent rectangle for ball control parameters
    def draw_team_ball_control(self,frame, frame_num,team_ball_control):
        overlay=frame.copy()
        cv2.rectangle(overlay,(1350,850),(1900,970),(255,255,255),-1)
        alpha=0.4  #for transparency
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        #put text
        team_ball_control_till_frame=team_ball_control[:frame_num+1] #ball control till that particular frame

        team1_num=team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team2_num=team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]

        team1=(team1_num/(team1_num+team2_num))*100
        team2=(team2_num/(team1_num+team2_num))*100

        cv2.putText(frame, f"TEAM 1 :{team1:.2f}%",(1400,900),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        cv2.putText(frame, f"TEAM 2 :{team2:.2f}%",(1400,950),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

        return frame


    #draw  trackers

    def draw_annotations(self, video_frames,tracks, team_ball_control):
        output_video_frames=[]
        for frame_num , frame in enumerate(video_frames):
            frame=frame.copy()  #copy the frame so we dont use original frame

            player_dict=tracks["player"][frame_num]
            ball_dict=tracks["ball"][frame_num]
            referee_dict=tracks["referee"][frame_num]

            for track_id, player in player_dict.items():
                color=player.get('team_color',(255,0,0))
                frame=self.draw_ellipse(frame, player["bbox"],color, track_id)
                if player.get('has_ball', False):
                    frame=self.draw_triangle(frame, player['bbox'], (0,0,255))

            for track_id, player in referee_dict.items():
                frame=self.draw_ellipse(frame, player["bbox"], (0,0,0), None)
            #Draw triangle on ball
            for track_id, ball in ball_dict.items():
                frame=self.draw_triangle(frame, ball["bbox"], (0,255,0))

            #draw team ball control

            frame=self.draw_team_ball_control(frame, frame_num,team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames