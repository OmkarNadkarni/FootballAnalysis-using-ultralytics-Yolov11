import pickle
import cv2
import numpy as np
import os
def measure_xy_distance(p1,p2):
    return p1[0] - p2[0], p1[1]- p2[1]

def measure_distance(p1,p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
class CameraMovementEstimator:
    def __init__(self, init_frame):
        self.file_path = r"../stubs/saved_camera_movement.pkl"
        self.file_loaded = False
        self.minDistance = 5
        self.camera_movement = []
        self.first_frame_grayScale = cv2.cvtColor(init_frame,cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(self.first_frame_grayScale)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1
        self.features = dict(
            maxCorners = 100,
            qualityLevel  = 0.3,
            minDistance = 3,
            blockSize = 7,
            mask = mask_features
        )

        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
        )


    def get_camera_movement(self, frames):

        if self.file_path is not None and os.path.exists(self.file_path) and not self.file_loaded:
            with open(self.file_path, 'rb') as f:
                self.file_loaded = True
                return pickle.load(f)
        # read from file
        camera_movement = [[0,0] for _ in  range(len(frames))]
        prev_grayScaleImg = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        prev_features = cv2.goodFeaturesToTrack(prev_grayScaleImg, **self.features)

        for frame_num in range(1, len(frames)):
            gray_img = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features,_,_ = cv2.calcOpticalFlowPyrLK(prev_grayScaleImg, gray_img, prev_features, None,  **self.lk_params)
            max_movement = 0
            camera_movement_x, camera_movement_y = 0,0
            for i,(newF, oldF) in enumerate(zip(new_features,prev_features)):
                new_feature_points = newF.ravel()
                old_feature_points = oldF.ravel()

                distance = measure_distance(new_feature_points, old_feature_points)
                if distance > max_movement:
                    max_movement = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_feature_points, new_feature_points)

            if max_movement > self.minDistance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_feature_points = cv2.goodFeaturesToTrack(gray_img, **self.features)

            prev_grayScaleImg = gray_img.copy()

        if not os.path.exists(self.file_path):
            with open(self.file_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement


def __main__():

    inputVideoFile = r"../inputVideo\video.mp4"
    cap = cv2.VideoCapture(inputVideoFile)
    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            cv2.waitKey(1)
        else:
            print("finished reading video")
            break
    cameraMovement = CameraMovementEstimator(frames[0])
    retVal = cameraMovement.get_camera_movement(frames)

if __name__ == "__main__":
    __main__()

