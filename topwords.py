import json
import os
import shutil

JSON_PATH = 'WLASL_v0.3.json'
SOURCE_VIDEOS = 'C:/Users/16sha/Downloads/videos/'
TARGET_DIR = 'WLASL100_Organized/'
NUM_WORDS = 100

with open(JSON_PATH, 'r') as f:
    data = json.load(f)

if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

for entry in data[:NUM_WORDS]:
    gloss = entry['gloss']
    instances = entry['instances']
    
    word_folder = os.path.join(TARGET_DIR, gloss)
    os.makedirs(word_folder, exist_ok=True)
    

    for inst in instances:
        video_id = inst['video_id']
        file_name = f"{video_id}.mp4"
        src_path = os.path.join(SOURCE_VIDEOS, file_name)
        dst_path = os.path.join(word_folder, file_name)

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"  Missing: {file_name}")

print("\nFinished! Your sorted dataset is in:", TARGET_DIR)