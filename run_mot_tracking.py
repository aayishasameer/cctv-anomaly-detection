from ultralytics import YOLO
import os

model = YOLO("yolov8n.pt")

VIDEO_DIR = r"C:\Users\aayis\Desktop\mainproject\cleaned_data\kaggle\working\normal_shop"
OUTPUT_DIR = "data/processed/tracks"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for video in os.listdir(VIDEO_DIR):
    if video.endswith(".mp4"):
        video_path = os.path.join(VIDEO_DIR, video)

        model.track(
            source=video_path,
            tracker="botsort.yaml",
            persist=True,
            classes=[0],      # person only (list is safer)
            conf=0.3,
            save=False,
            save_txt=True,
            project=OUTPUT_DIR,
            name=os.path.splitext(video)[0]
        )


print("Tracking complete!")
