from ultralytics import YOLO
import supervision as sv
import os
import pickle
import torch

class Detector:
    def __init__(self, model_path, saved_tracks_path):
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 0
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.file_path = saved_tracks_path
        self.file_loaded = False
        self.tracks = {
            "player": []
            #  FRAME1 : { track_id: {"bbox" : [0,0,0,0]}, track_id: {"bbox" : [0,0,0,0]} }
            #  FRAME2 : {track_id: {"bbox" : [0,0,0,0]}, track_id: {"bbox" : [0,0,0,0]}}
        }

    def predict_frame(self, frame):
        results = self.model.predict(frame, conf=0.5,device=self.device)
        return results

    def get_object_tracks(self,frame, frame_num):
        if self.file_path is not None and os.path.exists(self.file_path) and not self.file_loaded:
            with open(self.file_path, 'rb') as f:
                self.tracks = pickle.load(f)
                self.file_loaded = True

        if self.tracks["player"] and self.file_loaded:
            return self.tracks["player"][frame_num]


        detection_per_frame = self.predict_frame(frame)
        detections_to_save = []
        if detection_per_frame:
            for detection in detection_per_frame:
                for attrs in detection.boxes:
                    if attrs.cls == 0:  # only want to track humans for now
                        detection_supervision = sv.Detections.from_ultralytics(detection)
                        det_with_tracks = self.tracker.update_with_detections(detection_supervision)
                        self.tracks["player"].append({})
                        # print(det_with_tracks)
                        for frame_det in det_with_tracks:
                            bbox = frame_det[0].tolist()
                            track_id = frame_det[4]
                            self.tracks["player"][frame_num][track_id] = {"bbox" : bbox}

        return self.tracks["player"][frame_num]


    def get_all_tracks(self):
        return self.tracks

    def save_track_results(self):
        if self.file_path is not None and not os.path.exists(self.file_path):
            with open(self.file_path, 'wb') as f:
                pickle.dump(self.tracks, f)




