import cv2 as cv
import numpy as np
from ultralytics import YOLO
import cvzone
import time
import math
from sort import *

class ObjectDetector:
    def __init__(self, video_path, model_path, class_names):
        self.cap = cv.VideoCapture(video_path)
        self.model = YOLO(model_path)
        self.class_names = class_names
        self.my_color = (0, 0, 255)
        self.p_time = 0
        
        # Tracking setup
        self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        self.tracked_objects = {}  # Dictionary to store tracked object IDs and their last positions

    def process_frame(self, img):
        results = self.model(img, stream=True)
        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                current_class = self.class_names[cls]

                if conf > 0.5:  # Adjust confidence threshold as needed
                    cvzone.putTextRect(img, f'{current_class} {conf}', 
                                       (max(0, x1), max(35, y1)), scale=1, thickness=1, 
                                       colorB=self.my_color, colorT=(255, 255, 255), 
                                       colorR=self.my_color, offset=5)
                    cv.rectangle(img, (x1, y1), (x2, y2), self.my_color, 3)
                    
                    # Collect detections
                    current_array = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, current_array))

        # # Update tracker with detections
        # results_tracker = self.tracker.update(detections)

        # # Count objects based on their tracked IDs
        # for result in results_tracker:
        #     x1, y1, x2, y2, obj_id = result
        #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #     cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2  
        #     cv.circle(img, (cx, cy), 5, (255, 0, 255), cv.FILLED)  

        #     # Update tracked object positions
        #     self.tracked_objects[obj_id] = (cx, cy)
            
        #     # Optionally, apply a more complex counting logic here
        #     # For example, you might define regions dynamically or check for reappearance

        # # Display counts for each class
        # for class_name in self.class_names:
        #     # Use appropriate logic to count objects dynamically
        #     # For now, just displaying total tracked objects count
        #     cvzone.putTextRect(img, f'{class_name} Count: {len(self.tracked_objects)}', 
        #                        (50, 50 + 30 * self.class_names.index(class_name)),
        #                        scale=1, thickness=2, offset=10)

        return img

    def run(self):
        while True:
            success, img = self.cap.read()
            if not success:
                print("Frame not read")
                break

            img = self.process_frame(img)

            c_time = time.time()
            fps = 1 / (c_time - self.p_time) if self.p_time != 0 else 0
            self.p_time = c_time

            cv.putText(img, f'FPS: {int(fps)}', (10, 30), fontScale=1, 
                        fontFace=cv.FONT_HERSHEY_SCRIPT_SIMPLEX, color=(10, 10, 10), thickness=2)

            cv.imshow('Detector', img)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv.destroyAllWindows()

if __name__ == '__main__':
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

    video_path = "/Users/aaditya/ALSTOM/Camera-feed-prompting/Main/IMG_2182.mp4"
    model_path = "/Users/aaditya/ALSTOM/Camera-feed-prompting/YOLO/YOLO-Weights/yolov8n.pt"

    detector = ObjectDetector(video_path, model_path, class_names)
    detector.run()
