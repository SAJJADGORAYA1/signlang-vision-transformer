import os
import cv2
import mediapipe as mp
import numpy as np
import math
from torch.utils.data import Dataset

# Config (copied here for SignDataset dependency)
MAX_FRAMES = 30
KEYPOINT_DIM = (33 * 4) + (21 * 3) + (21 * 3) + (468 * 3)  # 1662

class KeypointExtractor:
    def __init__(self):
        self.holistic = mp.solutions.holistic.Holistic(static_image_mode=False)
    
    def extract(self, frame):
        results = self.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        def get_landmarks(landmark_list, expected_len, dims=4):
            if landmark_list:
                data = []
                for lm in landmark_list.landmark:
                    if dims == 4:
                        data.extend([lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else 1.0])
                    else:
                        data.extend([lm.x, lm.y, lm.z])
                if len(data) < expected_len:
                    data.extend([0.0] * (expected_len - len(data)))
                return data
            else:
                return [0.0] * expected_len
        
        pose = get_landmarks(results.pose_landmarks, 33 * 4, dims=4)
        left_hand = get_landmarks(results.left_hand_landmarks, 21 * 3, dims=3)
        right_hand = get_landmarks(results.right_hand_landmarks, 21 * 3, dims=3)
        face = get_landmarks(results.face_landmarks, 468 * 3, dims=3)
        
        keypoints = pose + left_hand + right_hand + face
        return np.array(keypoints, dtype=np.float32)

class SignDataset(Dataset):
    def __init__(self, video_dir):
        self.video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) 
                            if f.endswith(".mp4")]
        self.labels = [os.path.splitext(os.path.basename(v))[0] for v in self.video_paths]
        self.encoder = LabelEncoder()
        self.encoded_labels = self.encoder.fit_transform(self.labels)
        self.extractor = KeypointExtractor()
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.video_paths[idx])
        frames = []
        while len(frames) < MAX_FRAMES and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            keypoints = self.extractor.extract(frame)
            frames.append(keypoints)
        cap.release()
        
        if len(frames) < MAX_FRAMES:
            padding = [np.zeros(KEYPOINT_DIM, dtype=np.float32) 
                      for _ in range(MAX_FRAMES - len(frames))]
            frames.extend(padding)
        frames = np.stack(frames)
        label = self.encoded_labels[idx]
        return torch.tensor(frames, dtype=torch.float32), torch.tensor(label, dtype=torch.long)