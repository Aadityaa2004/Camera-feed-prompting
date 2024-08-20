import os
import base64  
import uuid
import json
import faiss
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from os import environ
from ultralytics import YOLO
from flask_migrate import Migrate
from sentence_transformers import SentenceTransformer
from sort import *
import math
from werkzeug.utils import secure_filename # Add the necessary import statements at the beginning of your file
from datetime import datetime
import pytz
from llama_cpp import Llama

app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = environ.get('DATABASE_URL', 'postgresql://aaditya@localhost:5431/aaditya')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Initialize YOLOv8 model
model_path = "/Users/aaditya/ALSTOM/Camera-feed-prompting/yolov8n.pt"
yolov8 = YOLO(model_path)

# Path to the directory where the model is saved
save_directory = "/Users/aaditya/dev/all-miniLM-L6-v2"

# Initialize SentenceTransformer model
model = SentenceTransformer(save_directory)

class ObjectDetection(db.Model):
    __tablename__ = 'object_detection'
    id = db.Column(db.String(36), primary_key=True)
    device_id = db.Column(db.String(100), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    json_data = db.Column(db.JSON)
    index_file = db.Column(db.LargeBinary)
    image_files = db.Column(db.JSON)  # Store image file paths as JSON
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  # Timestamp field

    def json(self):
        utc_time = self.created_at
        ist_time = utc_time.astimezone(pytz.timezone('Asia/Kolkata'))
        return {
            'id': self.id,
            'device_id': self.device_id,
            'file_path': self.file_path,
            'json_data': self.json_data,
            'image_files': self.image_files,
            'timestamp': ist_time.isoformat()  # Convert timestamp to IST ISO format
        }

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/push', methods=['POST'])
def push_data():
    data = request.json
    device_id = data.get('device_id')
    file_path = data.get('file_path')

    if not device_id or not file_path:
        return jsonify({'error': 'Device ID or file path not provided'}), 400

    if not os.path.isfile(file_path):
        return jsonify({'error': f'File not found: {file_path}'}), 404

    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension not in ['.jpg', '.jpeg', '.png', '.mp4']:
        return jsonify({'error': f'Unsupported file format: {file_path}'}), 400

    detection_id = str(uuid.uuid4())
    detection = ObjectDetection(id=detection_id, device_id=device_id, file_path=file_path)

    try:
        db.session.add(detection)
        db.session.commit()
        # Fetch the newly created record including the timestamp
        new_detection = ObjectDetection.query.get(detection_id)
        return jsonify({
            'message': 'Data pushed successfully',
            'id': detection_id,
            'timestamp': new_detection.created_at.isoformat()  # Include the timestamp
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/getfile', methods=['GET'])
def get_all_files():
    detections = ObjectDetection.query.all()
    files = [det.json() for det in detections]  # Use the updated json method
    return jsonify(files)

@app.route('/getfile/<device_id>', methods=['GET'])
def getfile(device_id):
    # Retrieve the record by device_id
    detection_record = ObjectDetection.query.filter_by(device_id=device_id).first()
    
    if not detection_record:
        return jsonify({'error': 'Device ID not found'}), 404
    
    file_path = detection_record.file_path
    
    if not os.path.isfile(file_path):
        return jsonify({'error': f'File not found: {file_path}'}), 404

    # Define device-specific directory and file paths
    device_dir = f'device_data_{device_id}'
    json_file_path = os.path.join(device_dir, 'detections.json')
    index_file_path = os.path.join(device_dir, 'faiss_index.index')
    images_dir = os.path.join(device_dir, 'tracked_images')

    # Check if we can return existing data
    if os.path.isdir(device_dir):
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            if os.path.isfile(json_file_path) and os.path.isfile(index_file_path):
                with open(json_file_path, 'r') as f:
                    detections = json.load(f)
                return jsonify({'message': 'Existing detection results found', 'detections': detections['detections']})
        elif file_path.lower().endswith('.mp4'):
            if os.path.isfile(json_file_path) and os.path.isfile(index_file_path) and os.path.isdir(images_dir):
                with open(json_file_path, 'r') as f:
                    detections = json.load(f)
                return jsonify({'message': 'Existing detection results found', 'detections': detections['detections']})

    # If we reach here, we need to process the file

    # Create device-specific directory and subdirectories
    os.makedirs(device_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    detections = []
    frame_index = 0
    first_frames = {}

    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Perform YOLO detection on image
        img = cv2.imread(file_path)
        results = yolov8(img, stream=True)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = yolov8.names[cls]

                detections.append({
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "confidence": conf,
                    "label": label,
                    "track_id": frame_index,
                    "frame_index": frame_index
                })
                frame_index += 1
    
    elif file_path.lower().endswith('.mp4'):
        confidence_threshold = 0.5
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return jsonify({'error': 'Could not open video file'}), 500

        mot_tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = yolov8(frame, stream=True)
            frame_detections = []

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = yolov8.names[cls]

                    frame_detections.append([x1, y1, x2, y2, conf])

            frame_detections = np.array(frame_detections)
            
            if len(frame_detections) > 0:
                trackers = mot_tracker.update(frame_detections)
            else:
                trackers = np.empty((0, 5))

            for trk in trackers:
                x1, y1, x2, y2, track_id = trk
                track_id = int(track_id)
                
                if track_id not in first_frames:
                    first_frames[track_id] = {
                        'frame': frame.copy(),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'label': label,
                        'conf': conf,
                        'frame_index': frame_index
                    }

                detections.append({
                    "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(conf),
                    "label": label,
                    "track_id": track_id,
                    "frame_index": frame_index
                })
            frame_index += 1

        cap.release()

        # Save PNG for each unique track_id
        for track_id, data in first_frames.items():
            frame = data['frame']
            x1, y1, x2, y2 = data['bbox']
            label = data['label']
            conf = data['conf']
            frame_index = data['frame_index']

            color = (0, 255, 0)
            label_text = f"{label} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            image_file_path = os.path.join(images_dir, f'frame_{frame_index}_track_{track_id:04d}.png')
            cv2.imwrite(image_file_path, frame)

    # Process detections to get unique track_ids
    unique_detections = {}

    for detection in detections:
        track_id = detection['track_id']
        
        if track_id not in unique_detections:
            unique_detections[track_id] = {
                "bbox_xyxy": detection['bbox_xyxy'],
                "confidence": detection['confidence'],
                "first_frame": detection['frame_index'],
                "label": detection['label'],
                "track_id": track_id,
                "frame_indexes": [detection['frame_index']]
            }
        else:
            unique_detections[track_id]['frame_indexes'].append(detection['frame_index'])

    final_detections = list(unique_detections.values())

    # Save detections to a JSON file
    with open(json_file_path, 'w') as f:
        json.dump({'detections': final_detections}, f, indent=2)

    # Create embeddings and update FAISS index
    labels = [det['label'] for det in final_detections]
    if not labels:
        return jsonify({'error': 'No valid labels to encode'}), 400

    print("Labels:", labels)

    try:
        embeddings = model.encode(labels)
        embeddings = embeddings.astype('float32')
    except Exception as e:
        return jsonify({'error': f'Error encoding labels: {str(e)}'}), 500

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    index.add(embeddings)
    faiss.write_index(index, index_file_path)

    return jsonify({'message': 'Detection results and embeddings updated successfully', 'detections': final_detections})

@app.route('/getfile/<device_id>/query/<user_input>', methods=['GET'])
def query_file(device_id, user_input):
    # Retrieve the record by device_id
    detection_record = ObjectDetection.query.filter_by(device_id=device_id).first()
    
    if not detection_record:
        return jsonify({'error': 'Device ID not found'}), 404
    
    file_path = detection_record.file_path
    
    if not os.path.isfile(file_path):
        return jsonify({'error': f'File not found: {file_path}'}), 404

    # Define device-specific directory and file paths
    device_dir = f'device_data_{device_id}'
    json_file_path = os.path.join(device_dir, 'detections.json')
    index_file_path = os.path.join(device_dir, 'faiss_index.index')
    images_dir = os.path.join(device_dir, 'tracked_images')

    # Check if necessary files exist
    if not os.path.isfile(json_file_path) or not os.path.isfile(index_file_path):
        return jsonify({'error': 'Detection or index file not found'}), 404

    # Convert user input to embeddingZ
    query_embedding = model.encode([user_input]).astype('float32')

    # Load FAISS index and perform search
    index = faiss.read_index(index_file_path)
    D, I = index.search(query_embedding, k=20)  # Search for top 10 matches
    
    # Load detection results
    with open(json_file_path, 'r') as f:
        detection_data = json.load(f)
    
    results = []
    for i, idx in enumerate(I[0]):
        if idx < len(detection_data['detections']) and D[0][i] < 0.8:  # Threshold for similarity
            detection = detection_data['detections'][idx]
            detection['distance'] = float(D[0][i])
            results.append(detection)
    
    if results:
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Handle image file
            image = cv2.imread(file_path)
            for result in results:
                bbox = result['bbox_xyxy']
                label = result['label']
                confidence = result['confidence']
                
                # Draw bounding box
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                label_text = f"{label}: {confidence:.2f}"
                cv2.putText(image, label_text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Convert image to base64
            _, buffer = cv2.imencode('.png', image)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'message': 'Image annotated and encoded',
                'image_base64': f'data:image/png;base64,{base64_image}',
                'results': results
            })
        elif file_path.lower().endswith('.mp4'):
            # Handle video file
            relevant_frames = []
            
            for result in results:
                track_id = result['track_id']
                frame_index = result['first_frame']
                image_file_path = os.path.join(images_dir, f'frame_{frame_index}_track_{track_id:04d}.png')
                
                if os.path.isfile(image_file_path):
                    image = cv2.imread(image_file_path)
                    _, buffer = cv2.imencode('.png', image)
                    base64_image = base64.b64encode(buffer).decode('utf-8')
                    relevant_frames.append({
                        'frame_index': frame_index,
                        'track_id': track_id,
                        'image_base64': f'data:image/png;base64,{base64_image}'
                    })
            
            return jsonify({
                'message': 'Relevant video frames found',
                'relevant_frames': relevant_frames,
                'results': results
            })
    else:
        return jsonify({'message': 'No similar objects found', 'results': results})


@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Welcome to the object detection API'}), 200

@app.route('/delete/<device_id>', methods=['DELETE'])
def delete_by_device_id(device_id):
    if not device_id:
        return jsonify({'error': 'Device ID not provided'}), 400

    try:
        # Delete associated files from the file system
        device_dir = f'device_data_{device_id}'
        
        if os.path.isdir(device_dir):
            # Delete JSON file
            json_file_path = os.path.join(device_dir, 'detections.json')
            if os.path.isfile(json_file_path):
                os.remove(json_file_path)

            # Delete index file
            index_file_path = os.path.join(device_dir, 'faiss_index.index')
            if os.path.isfile(index_file_path):
                os.remove(index_file_path)

            # Delete tracked images
            images_dir = os.path.join(device_dir, 'tracked_images')
            if os.path.isdir(images_dir):
                for file in os.listdir(images_dir):
                    file_path = os.path.join(images_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                os.rmdir(images_dir)

            # Remove the device directory
            os.rmdir(device_dir)
        
        # Retrieve records with the specified device_id
        detections = ObjectDetection.query.filter_by(device_id=device_id).all()
        
        if not detections:
            return jsonify({'message': 'No records found for the given device ID'}), 404
        
        # Delete related files
        for detection in detections:
            file_path = detection.file_path
            json_file_path = f'detections_{device_id}.json'
            index_file_path = 'my_faiss_index.index'
            annotated_image_path = f'annotated_{device_id}.png'
            
            if os.path.isfile(json_file_path):
                os.remove(json_file_path)
            if os.path.isfile(index_file_path):
                os.remove(index_file_path)
            if os.path.isfile(annotated_image_path):
                os.remove(annotated_image_path)
        
        # Delete records from the database
        num_deleted = ObjectDetection.query.filter_by(device_id=device_id).delete(synchronize_session=False)
        db.session.commit()

        return jsonify({'message': f'{num_deleted} records and associated files deleted successfully'}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Error deleting data: {str(e)}'}), 500

    

# Directory where annotated images are stored
ANNOTATED_IMAGES_DIR = '/path/to/annotated_images'

# Add a new route to serve annotated images
@app.route('/annotated_images/<filename>')
def serve_image(filename):
    return send_from_directory(ANNOTATED_IMAGES_DIR, filename)

@app.route('/delete_device_data/<device_id>', methods=['DELETE'])
def delete_device_data(device_id):
    device_dir = f'device_data_{device_id}'
    
    if not os.path.isdir(device_dir):
        return jsonify({'message': f'No data found for device_id: {device_id}'}), 404

    try:
        # Delete JSON file
        json_file_path = os.path.join(device_dir, 'detections.json')
        if os.path.isfile(json_file_path):
            os.remove(json_file_path)

        # Delete index file
        index_file_path = os.path.join(device_dir, 'faiss_index.index')
        if os.path.isfile(index_file_path):
            os.remove(index_file_path)

        # Delete tracked images
        images_dir = os.path.join(device_dir, 'tracked_images')
        if os.path.isdir(images_dir):
            for file in os.listdir(images_dir):
                file_path = os.path.join(images_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            os.rmdir(images_dir)

        # Remove the device directory
        os.rmdir(device_dir)

        return jsonify({'message': f'All data for device_id {device_id} has been deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': f'Error deleting data: {str(e)}'}), 500
    
@app.route('/show_history', methods=['GET'])
def show_history():
    base_directory = '.'  # Adjust this to the actual base directory where device data is stored
    device_dirs = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d)) and d.startswith('device_data_')]

    results = []

    for device_dir in device_dirs:
        device_id = device_dir[len('device_data_'):]
        device_path = os.path.join(base_directory, device_dir)

        # Look up the database to get the file_path associated with this device_id
        detection_record = ObjectDetection.query.filter_by(device_id=device_id).first()

        if detection_record:
            file_path = detection_record.file_path
            json_file_path = os.path.join(device_path, 'detections.json')
            index_file_path = os.path.join(device_path, 'faiss_index.index')
            images_dir = os.path.join(device_path, 'tracked_images')

            # Collect data from the folder
            folder_data = {
                'device_id': device_id,
                'file_path': file_path,
                'json_file_exists': os.path.isfile(json_file_path),
                'index_file_exists': os.path.isfile(index_file_path),
                'images_count': len(os.listdir(images_dir)) if os.path.isdir(images_dir) else 0
            }

            results.append(folder_data)

    return jsonify({'devices': results})


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

