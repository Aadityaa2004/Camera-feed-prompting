import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import cv2
import os

# Load the model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Sample JSON data
json_data = [
[
    {
        "detections": [
            {
                "bbox_xyxy": [
                    59,
                    317,
                    283,
                    541
                ],
                "confidence": 0.6857069134712219,
                "label": "chair"
            },
            {
                "bbox_xyxy": [
                    157,
                    303,
                    300,
                    426
                ],
                "confidence": 0.6600837111473083,
                "label": "couch"
            },
            {
                "bbox_xyxy": [
                    517,
                    339,
                    693,
                    541
                ],
                "confidence": 0.557014524936676,
                "label": "chair"
            },
            {
                "bbox_xyxy": [
                    357,
                    214,
                    435,
                    261
                ],
                "confidence": 0.5541404485702515,
                "label": "tv"
            },
            {
                "bbox_xyxy": [
                    586,
                    322,
                    647,
                    388
                ],
                "confidence": 0.5526561737060547,
                "label": "potted plant"
            },
            {
                "bbox_xyxy": [
                    478,
                    310,
                    591,
                    418
                ],
                "confidence": 0.31353524327278137,
                "label": "couch"
            }
        ],
        "device_id": "camera_02",
        "file_name": "house interior.jpg",
        "file_path": "/Users/aaditya/ALSTOM/Camera-feed-prompting/images/house interior.jpg"
    },
    {
        "detections": [
            {
                "bbox_xyxy": [
                    101,
                    34,
                    175,
                    252
                ],
                "confidence": 0.9297449588775635,
                "label": "person"
            },
            {
                "bbox_xyxy": [
                    279,
                    164,
                    349,
                    262
                ],
                "confidence": 0.9113397598266602,
                "label": "dog"
            },
            {
                "bbox_xyxy": [
                    176,
                    48,
                    255,
                    253
                ],
                "confidence": 0.9098949432373047,
                "label": "person"
            }
        ],
        "device_id": "camera_02",
        "file_name": "people walking.png",
        "file_path": "/Users/aaditya/ALSTOM/Camera-feed-prompting/images/people walking.png"
    },
    {
        "detections": [
            {
                "bbox_xyxy": [
                    1103,
                    750,
                    1262,
                    883
                ],
                "confidence": 0.8417739272117615,
                "label": "car"
            },
            {
                "bbox_xyxy": [
                    832,
                    529,
                    956,
                    630
                ],
                "confidence": 0.7874930500984192,
                "label": "car"
            },
            {
                "bbox_xyxy": [
                    820,
                    649,
                    966,
                    799
                ],
                "confidence": 0.7742293477058411,
                "label": "car"
            },
            {
                "bbox_xyxy": [
                    1056,
                    588,
                    1186,
                    712
                ],
                "confidence": 0.7019590139389038,
                "label": "car"
            },
            {
                "bbox_xyxy": [
                    1124,
                    892,
                    1296,
                    1063
                ],
                "confidence": 0.6582618951797485,
                "label": "car"
            },
            {
                "bbox_xyxy": [
                    970,
                    188,
                    1054,
                    266
                ],
                "confidence": 0.5820863246917725,
                "label": "car"
            },
            {
                "bbox_xyxy": [
                    513,
                    447,
                    743,
                    782
                ],
                "confidence": 0.5246092081069946,
                "label": "bus"
            },
            {
                "bbox_xyxy": [
                    822,
                    171,
                    905,
                    236
                ],
                "confidence": 0.445484459400177,
                "label": "car"
            },
            {
                "bbox_xyxy": [
                    821,
                    807,
                    1028,
                    982
                ],
                "confidence": 0.4330670237541199,
                "label": "car"
            },
            {
                "bbox_xyxy": [
                    629,
                    334,
                    748,
                    450
                ],
                "confidence": 0.3801348805427551,
                "label": "car"
            },
            {
                "bbox_xyxy": [
                    1037,
                    124,
                    1130,
                    173
                ],
                "confidence": 0.28444230556488037,
                "label": "traffic light"
            }
        ],
        "device_id": "camera_02",
        "file_name": "highway.jpg",
        "file_path": "/Users/aaditya/ALSTOM/Camera-feed-prompting/images/highway.jpg"
    }
]
    # Add more entries as needed
]



# Extract labels and create embeddings
labels = []
label_data = []
for entry in json_data[0]:  # Access the first (and only) element of the outer list
    for detection in entry['detections']:
        labels.append(detection['label'])
        label_data.append({
            'label': detection['label'],
            'bbox': detection['bbox_xyxy'],
            'device_id': entry['device_id'],
            'file_name': entry['file_name'],
            'file_path': entry['file_path'],
            'confidence': detection['confidence']
        })

label_embeddings = model.encode(labels).astype('float32')

# Create and populate the FAISS index
dimension = label_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(label_embeddings)

# Save the FAISS index
faiss.write_index(index, 'my_faiss_index.index')

# Load the FAISS index
loaded_index = faiss.read_index('my_faiss_index.index')

def search_query(query_text):
    # Encode the query
    query_embedding = model.encode([query_text]).astype('float32')
    
    # Search the index
    D, I = loaded_index.search(query_embedding, k=len(labels))  # Search all labels
    
    results = []
    for i, idx in enumerate(I[0]):
        if D[0][i] < 0.5:  # Threshold for similarity
            result = label_data[idx]
            result['distance'] = float(D[0][i])  # Add distance to results
            results.append(result)
    
    return results

def plot_bounding_boxes(results):
    if not results:
        print("No results to plot.")
        return

    # Get the first result's file path (assuming all results are from the same image)
    file_path = results[0]['file_path']
    
    # Read the image
    image = cv2.imread(file_path)
    if image is None:
        print(f"Unable to read image from {file_path}")
        return

    # Plot bounding boxes
    for result in results:
        bbox = result['bbox']
        label = result['label']
        confidence = result['confidence']
        
        # Convert bbox to integers
        bbox = [int(coord) for coord in bbox]
        
        # Draw bounding box
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        # Prepare label text
        label_text = f"{label}: {confidence:.2f}"
        
        # Put label text
        cv2.putText(image, label_text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save the image with bounding boxes
    output_path = 'output_' + os.path.basename(file_path)
    cv2.imwrite(output_path, image)
    print(f"Image with bounding boxes saved as {output_path}")

def handle_user_input():
    user_query = input("Enter your query: ")
    results = search_query(user_query)
    
    if results:
        for result in results:
            print(f"Label: {result['label']}")
            print(f"Bounding Box: {result['bbox']}")
            print(f"Device ID: {result['device_id']}")
            print(f"File Name: {result['file_name']}")
            print(f"File Path: {result['file_path']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Distance: {result['distance']}")
            print()
        
        # Plot bounding boxes
        plot_bounding_boxes(results)
    else:
        print("No results found.")

# Run the input handling
if __name__ == '__main__':
    handle_user_input()

