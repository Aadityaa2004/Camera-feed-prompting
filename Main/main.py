import cv2 as cv
import numpy as np
from ultralytics import YOLO
import cvzone
import time
import math
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
import json

app = FastAPI()

class DetectedObject(BaseModel):
    class_name: str
    confidence: float
    bbox: List[int]

class FrameResult(BaseModel):
    timestamp: float
    objects: List[DetectedObject]

class ObjectDetector:
    def __init__(self, video_path, model_path, class_names):
        self.cap = cv.VideoCapture(video_path)
        self.model = YOLO(model_path)
        self.class_names = class_names
        self.my_color = (0, 0, 255)
        self.p_time = 0

    def process_frame(self, img):
        results = self.model(img, stream=True)
        detections = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                current_class = self.class_names[cls]

                if conf > 0.5:  # Adjust confidence threshold as needed
                    detections.append(DetectedObject(
                        class_name=current_class,
                        confidence=conf,
                        bbox=[x1, y1, x2, y2]
                    ))

        return detections

    def run(self):
        frame_count = 0
        results = []
        output_dir = "video_output"
        os.makedirs(output_dir, exist_ok=True)

        while True:
            success, img = self.cap.read()
            if not success:
                break

            timestamp = self.cap.get(cv.CAP_PROP_POS_MSEC) / 1000.0
            detections = self.process_frame(img)

            frame_result = FrameResult(
                timestamp=timestamp,
                objects=detections
            )
            results.append(frame_result)

            # Save frame as image
            frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
            cv.imwrite(frame_filename, img)

            # Save frame result as JSON
            result_filename = os.path.join(output_dir, f"result_{frame_count:04d}.json")
            with open(result_filename, 'w') as f:
                json.dump(frame_result.dict(), f, indent=4)

            frame_count += 1

        self.cleanup()

        # Save overall results
        with open(os.path.join(output_dir, "all_results.json"), 'w') as f:
            json.dump([r.dict() for r in results], f, indent=4)

        return frame_count, output_dir

    def cleanup(self):
        self.cap.release()

@app.get("/process_video/")
async def process_video():
    video_path = "/Users/aaditya/ALSTOM/Camera-feed-prompting/Main/IMG_2182.mp4"
    model_path = "/Users/aaditya/ALSTOM/Camera-feed-prompting/YOLO/YOLO-Weights/yolov8n.pt"
    class_names = [
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
        "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"
    ]

    detector = ObjectDetector(video_path, model_path, class_names)
    total_frames, output_dir = detector.run()

    return {"total_frames": total_frames, "output_directory": output_dir}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)