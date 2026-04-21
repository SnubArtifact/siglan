import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

INPUT_DIR = 'WLASL100_Organized'
OUTPUT_DIR = 'WLASL100_Features_Slim'
MODEL_PATH = 'holistic_landmarker.task'

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HolisticLandmarkerOptions(base_options=base_options)

def extract_slim_features(frame, detector_instance):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    try:
        result = detector_instance.detect(mp_image)
    except:
        return np.zeros(258)

    pose = np.array([[l.x, l.y, l.z, l.visibility] for l in result.pose_landmarks]).flatten() if result.pose_landmarks else np.zeros(132)

    lh = np.array([[l.x, l.y, l.z] for l in result.left_hand_landmarks]).flatten() if result.left_hand_landmarks else np.zeros(63)
    rh = np.array([[l.x, l.y, l.z] for l in result.right_hand_landmarks]).flatten() if result.right_hand_landmarks else np.zeros(63)
    
    return np.concatenate([pose, lh, rh])

os.makedirs(OUTPUT_DIR, exist_ok=True)
words = sorted(os.listdir(INPUT_DIR))

for word in words:
    word_path = os.path.join(INPUT_DIR, word)
    if not os.path.isdir(word_path): continue
    
    out_word_path = os.path.join(OUTPUT_DIR, word)
    os.makedirs(out_word_path, exist_ok=True)
    
    print(f"\n>>> Word: {word.upper()}")
    
    for video_file in os.listdir(word_path):
        if not video_file.endswith('.mp4'): continue
        
        detector = vision.HolisticLandmarker.create_from_options(options)
        cap = cv2.VideoCapture(os.path.join(word_path, video_file))
        frames_list = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frames_list.append(extract_slim_features(frame, detector))
            
        cap.release()
        detector.close()
        
        if len(frames_list) > 0:
            arr = np.array(frames_list)
            indices = np.linspace(0, len(arr) - 1, 30).astype(int)
            standardized = arr[indices]
            
            motion_range = np.max(standardized) - np.min(standardized)
            
            if motion_range > 0.05:
                save_path = os.path.join(out_word_path, video_file.replace('.mp4', '.npy'))
                np.save(save_path, standardized)
                print(f"  [OK] {video_file} (Motion: {motion_range:.2f})")
            else:
                print(f"  [SKIP] {video_file} - No motion detected.")

print("\nExtraction Complete.")