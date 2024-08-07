import os
import base64  
import uuid
import json
import faiss
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from os import environ
from ultralytics import YOLO
from flask_migrate import Migrate
from sentence_transformers import SentenceTransformer

# Add the necessary import statements at the beginning of your file
from werkzeug.utils import secure_filename
app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = environ.get('DATABASE_URL', 'postgresql://aaditya@localhost:5431/aaditya')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Initialize YOLOv8 model
model_path = "/Users/aaditya/ALSTOM/Camera-feed-prompting/YOLO/YOLO-Weights/yolov8n.pt"
yolov8 = YOLO(model_path)

# Initialize SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

class ObjectDetection(db.Model):
    __tablename__ = 'object_detection'
    id = db.Column(db.String(36), primary_key=True)
    device_id = db.Column(db.String(100), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)

    def json(self):
        return {
            'id': self.id,
            'device_id': self.device_id,
            'file_path': self.file_path
        }


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({'message': 'File uploaded successfully', 'file_path': file_path}), 201
    else:
        return jsonify({'error': 'Invalid file type'}), 400


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
    if file_extension not in ['.jpg', '.jpeg', '.png']:
        return jsonify({'error': f'Unsupported file format: {file_path}'}), 400

    detection_id = str(uuid.uuid4())
    detection = ObjectDetection(id=detection_id, device_id=device_id, file_path=file_path)

    try:
        db.session.add(detection)
        db.session.commit()
        return jsonify({'message': 'Data pushed successfully', 'id': detection_id}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/getfile', methods=['GET'])
def get_all_files():
    detections = ObjectDetection.query.all()
    files = [{'device_id': det.device_id, 'file_path': det.file_path} for det in detections]
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

    # Define file paths
    json_file_path = f'detections_{device_id}.json'
    index_file_path = 'my_faiss_index.index'
    annotated_image_path = f'annotated_{device_id}.png'
    
    # Delete all JSON and PNG files in the current directory
    for file in os.listdir('.'):
        if file.endswith('.json') or file.endswith('.png'):
            os.remove(file)

    # Delete previously created index file if it exists
    if os.path.isfile(index_file_path):
        os.remove(index_file_path)
    
    # Perform YOLO detection
    img = cv2.imread(file_path)
    results = yolov8(img, stream=True)
    
    # Process results
    detections = []
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
                "label": label
            })
    
    # Save detections to a JSON file
    with open(json_file_path, 'w') as f:
        json.dump({'detections': detections}, f)

    # Create embeddings and update FAISS index
    labels = [det['label'] for det in detections]
    embeddings = model.encode(labels).astype('float32')
    
    # Create or load FAISS index
    if os.path.exists(index_file_path):
        index = faiss.read_index(index_file_path)
    else:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
    
    index.add(embeddings)
    faiss.write_index(index, index_file_path)

    return jsonify({'message': 'Detection results and embeddings updated successfully', 'detections': detections})


@app.route('/getfile/<device_id>/query/<user_input>', methods=['GET'])
def query_file(device_id, user_input):
    # Retrieve the record by device_id
    detection_record = ObjectDetection.query.filter_by(device_id=device_id).first()
    
    if not detection_record:
        return jsonify({'error': 'Device ID not found'}), 404
    
    file_path = detection_record.file_path
    
    if not os.path.isfile(file_path):
        return jsonify({'error': f'File not found: {file_path}'}), 404

    # Convert user input to embedding
    query_embedding = model.encode([user_input]).astype('float32')

    # Load FAISS index and perform search
    index_file = 'my_faiss_index.index'
    if not os.path.exists(index_file):
        return jsonify({'error': 'Index file not found'}), 404
    
    index = faiss.read_index(index_file)
    D, I = index.search(query_embedding, k=10)  # Search for top 10 matches
    
    # Load detection results
    json_file_path = f'detections_{device_id}.json'
    if not os.path.isfile(json_file_path):
        return jsonify({'error': 'Detection file not found'}), 404
    
    with open(json_file_path, 'r') as f:
        detection_data = json.load(f)
    
    results = []
    for i, idx in enumerate(I[0]):
        if D[0][i] < 0.5:  # Threshold for similarity
            detection = detection_data['detections'][idx]
            detection['distance'] = float(D[0][i])
            results.append(detection)
    
    # Plot bounding boxes if results are found
    if results:
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
        return jsonify({'error': str(e)}), 500

# Directory where annotated images are stored
ANNOTATED_IMAGES_DIR = '/path/to/annotated_images'


# Add a new route to serve annotated images
@app.route('/annotated_images/<filename>')
def serve_image(filename):
    return send_from_directory(ANNOTATED_IMAGES_DIR, filename)


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
