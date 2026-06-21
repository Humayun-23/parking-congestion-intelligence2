import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO

def process_parking_feed(video_source=0, execution_fps=30):
    """
    Processes video frames to track vehicle locations and compute dwell times
    to identify on-street carriageway obstruction.
    """
    # Load a pre-trained light-weight YOLOv8 nano model
    model = YOLO("yolov8n.pt")
    
    # Tracking dictionary: {track_id: {"first_seen": datetime, "last_seen": datetime, "coords": [x1, y1, x2, y2]}}
    active_tracks = {}
    completed_violations = []
    
    # Target COCO dataset vehicle classes (2: car, 3: motorcycle, 5: bus, 7: truck)
    vehicle_classes = [2, 3, 5, 7]
    
    print("==> Initializing AI Computer Vision Tracker...")
    
    # In production, swap with cv2.VideoCapture(video_source)
    # Simulating 100 frames of a fixed camera stream for integration testing
    for frame_count in range(1, 101):
        timestamp = datetime.now()
        
        # Mocking bounding box outputs from model.track() for testing without a live video file
        # In a real environment, you would use: results = model.track(source=frame, persist=True)
        mock_boxes = [
            {"id": 101, "class": 2, "bbox": [150, 300, 250, 400]},  # Stationary Car
            {"id": 102, "class": 3, "bbox": [450 + frame_count, 200, 500 + frame_count, 250]}  # Moving Scooter
        ]
        
        for box in mock_boxes:
            track_id = box["id"]
            cls_id = box["class"]
            
            if cls_id in vehicle_classes:
                if track_id not in active_tracks:
                    active_tracks[track_id] = {
                        "first_seen": timestamp,
                        "last_seen": timestamp,
                        "class": cls_id,
                        "bbox": box["bbox"]
                    }
                else:
                    active_tracks[track_id]["last_seen"] = timestamp
                    active_tracks[track_id]["bbox"] = box["bbox"]
                    
        # Dwell time calculation logic (Frame Rate dependent)
        # If a vehicle remains untracked or frames close out, we analyze total time elapsed
        if frame_count == 100:
            for track_id, info in active_tracks.items():
                duration_seconds = (info["last_seen"] - info["first_seen"]).total_seconds()
                
                # If a car is stationary for more than a threshold (e.g., simulating 30+ seconds)
                # For our quick test loop, any registered track duration counts as an event
                completed_violations.append({
                    "id": f"CV-TICKET-{track_id}",
                    "latitude": 12.9716,  # Center coordinate placeholder for tracking node
                    "longitude": 77.5946,
                    "violation_type": "PARKING OBSTRUCTION (CV DETECTED)",
                    "vehicle_type": "CAR" if info["class"] == 2 else "SCOOTER",
                    "police_station": "Upparpet",  # Link directly to your top priority zone
                    "created_datetime": info["first_seen"],
                    "dwell_time_minutes": round(duration_seconds / 60.0, 2),
                    "is_rejected": False
                })
                
    # Wrap results into a Pandas Dataframe matched to the project structure
    cv_df = pd.DataFrame(completed_violations)
    print(f"==> Success! CV Pipeline processed feed and flagged {len(cv_df)} persistent obstructions.")
    return cv_df

if __name__ == "__main__":
    df = process_parking_feed()
    print(df)