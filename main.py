import cv2
from ultralytics import YOLO
from Detector import Detector
import time
from TeamClustering import TeamClustering

inputVideoFile = r"inputVideo\video.mp4"
cap = cv2.VideoCapture(inputVideoFile)
tracker = Detector("yolo11n.pt", "stubs/saved_tracks.pkl")
team_cluster = TeamClustering()
frame_num = 0

# Define output video properties
output_filename = "output.mp4"
frame_width = 640
frame_height = 360
fps = 30

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'mp4v' for .mp4
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
pTime = time.time()
while True:
    ret, frame = cap.read()

    if ret:
        frame = cv2.resize(frame,(frame_width,frame_height))
        current_frame_tracks = tracker.get_object_tracks(frame, frame_num)
        team_cluster.assign_team_color(frame,current_frame_tracks)
        frame_num += 1
        for track_id, bbox in current_frame_tracks.items():
            xyxy = [int(val) for val in bbox["bbox"]]
            team = team_cluster.get_player_team(frame, bbox["bbox"], track_id)
            rect_color = (255,0,0)
            if team == 1:
                rect_color = (255, 0, 255)
            cv2.rectangle(frame,(xyxy[0],xyxy[1]), (xyxy[2], xyxy[3]),rect_color,2)
            cv2.putText(frame,str(track_id),(xyxy[0] - 5,xyxy[1] - 5),cv2.FONT_HERSHEY_SIMPLEX,0.4, (255,0,0))

        cTime = time.time()
        fps = int(1/(cTime - pTime))
        pTime = cTime
        cv2.putText(frame, str(fps), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0),2)
        cv2.imshow("Image", frame)
        out.write(frame)
        c = cv2.waitKey(1)

        if c == ord('q'):
            break
    else:
        out.release()
        print("file complete")
        tracker.save_track_results()
        break

