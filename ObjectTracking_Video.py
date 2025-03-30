#Object Detection and Tracking From a Video

import cv2
import numpy as np

# Load YOLO pre-trained weights and configuration files
yolo_weights = "yolov3.weights"
yolo_config = "yolov3.cfg"
labels_file = "coco.names"

# Load class names
with open(labels_file, "r") as f:
    class_names = f.read().strip().split("\n")

# Initialize the YOLO model
net = cv2.dnn.readNet(yolo_weights, yolo_config)

# Initialize video capture
video_path = ("project_video.mp4")  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Increase playback speed by skipping frames
playback_speed = 6

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames to increase speed
    frame_count += 1
    if frame_count % playback_speed != 0:
        continue

    height, width = frame.shape[:2]

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layer names
    layer_names = net.getUnconnectedOutLayersNames()

    # Run forward pass
    detections = net.forward(layer_names)

    # Initialize lists for NMS
    boxes = []
    confidences = []
    class_ids = []

    # Process detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_names[class_id] == "car": # Adjust confidence threshold as needed
                box = obj[0:4] * np.array([width, height, width, height])
                center_x, center_y, w, h = box.astype("int")

                # Calculate top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    # Draw the final bounding boxes
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Object Detection and Tracking", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()