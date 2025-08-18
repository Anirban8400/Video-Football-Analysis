import cv2
import numpy as np
from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None

    def get_clustering_model(self, image):
        # Flatten image to (num_pixels, 3) and cast to float32
        image_2d = image.reshape(-1, 3).astype(np.float32)
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        # bbox is [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, bbox)

        # Crop player region
        image = frame[y1:y2, x1:x2]

        # Use only top half (jersey area)
        top_half_image = image[0:int(image.shape[0]//2), :]

        # Get clustering model for top half
        kmeans = self.get_clustering_model(top_half_image)

        # Cluster labels reshaped to image dimensions
        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Detect background cluster from corner pixels
        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1]
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        # Return player's average jersey color as float32
        return kmeans.cluster_centers_[player_cluster].astype(np.float32)
    
    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, det in player_detections.items():
            bbox = det["bbox"]
            player_colors.append(self.get_player_color(frame, bbox))

        # Stack colors into (n_players, 3) array
        player_colors = np.vstack(player_colors).astype(np.float32)

        # Cluster into 2 teams
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10)
        kmeans.fit(player_colors)
        self.kmeans = kmeans

        # Store team colors
        self.team_colors[1] = np.array([255,  255, 255], dtype=np.float32)
        self.team_colors[2] = np.array([128,  128, 128], dtype=np.float32)

    def get_player_team(self, frame, player_bbox, player_id):
        # Return cached result if available
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        # Get player's jersey color
        player_color = self.get_player_color(frame, player_bbox)

        # Predict team and add 1 to get team IDs 1/2
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1

        # Cache the result
        self.player_team_dict[player_id] = team_id
        return team_id


