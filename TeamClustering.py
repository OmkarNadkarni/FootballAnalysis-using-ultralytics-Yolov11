from sklearn.cluster import KMeans
class TeamClustering:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {} # playerId : team1/team2
        self.kmeans = None
        pass

    def assign_team_color(self,frame, player_detections):
        player_colors = []
        for _, detection in player_detections.items():
            bbox = detection["bbox"]
            player_color = self.get_player_color(frame,bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters = 2, init="k-means++",n_init=1)
        kmeans.fit(player_colors)
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self,frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        player_color = self.get_player_color(frame,player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1,-1))
        # want the team as 1 or 2
        team_id +=1
        self.player_team_dict[player_id] = team_id
        return team_id
    def get_player_color(self,frame,bbox):
        img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        # only want the jersey so get top half of image.
        topHalfImage = img[0: int(img.shape[0]/2),:]
        kmeans = self.get_clustering_model(topHalfImage)

        #get labels
        labels = kmeans.labels_
        clustered_img = labels.reshape(topHalfImage.shape[0],topHalfImage.shape[1])

        # get player cluster
        corner_cluster = [clustered_img[0,0],clustered_img[0,-1],clustered_img[-1,0],clustered_img[-1,-1]]
        non_player_cluster = max(set(corner_cluster), key= corner_cluster.count)
        player_cluster = 1 - non_player_cluster
        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color

    def get_clustering_model(self,jersey_image):
        # reshape into 2D array
        image_2d = jersey_image.reshape(-1,3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)
        return kmeans


